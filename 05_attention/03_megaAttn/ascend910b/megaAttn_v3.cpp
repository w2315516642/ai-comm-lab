#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "megaAttn_v3.h"

using namespace AscendC;
using namespace mega_ascend_v3;

// 两个 AIV 准备同一份 GM 矩阵；IBSHARE=true 让 Cube 等两侧都写完再计算。
using AType=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,false,LayoutMode::NONE,true>;
using KType=MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t,true,LayoutMode::NONE,true>;
using CType=MatmulType<TPosition::GM,CubeFormat::ND,float>;

// 两个 AIV 各处理 32 head/32 行 KV；Cube 每次计算完整 64 head。
class KernelMegaAttentionV3 {
  public:
    Matmul<AType,KType,CType> qkMm;
    Matmul<AType,AType,CType> pvMm;

    __aicore__ inline void Init(
        GM_ADDR q,GM_ADDR kv8,GM_ADDR kv4,GM_ADDR ids,GM_ADDR extraIds,
        GM_ADDR rope,GM_ADDR sink,GM_ADDR configs,uint32_t count,
        GM_ADDR fp8,GM_ADDR scales,GM_ADDR lse,int32_t sfStride,
        GM_ADDR scratch,GM_ADDR profiles,GM_ADDR tables,TPipe *pipe
    ) {
        qGm.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        kv8Gm.SetGlobalBuffer((__gm__ uint8_t*)kv8);
        kv4Gm.SetGlobalBuffer((__gm__ uint8_t*)kv4);
        idsGm.SetGlobalBuffer((__gm__ int32_t*)ids);
        extraIdsGm.SetGlobalBuffer((__gm__ int32_t*)extraIds);
        ropeGm.SetGlobalBuffer((__gm__ float*)rope);
        hasSink=sink!=nullptr;
        if (hasSink) sinkGm.SetGlobalBuffer((__gm__ float*)sink);
        configsGm.SetGlobalBuffer((__gm__ int32_t*)configs);
        fp8Gm.SetGlobalBuffer((__gm__ uint8_t*)fp8);
        scalesGm.SetGlobalBuffer((__gm__ uint32_t*)scales);
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        profiling=profiles!=nullptr;
        if (profiling) profileGm.SetGlobalBuffer((__gm__ uint64_t*)profiles);
        taskCount=count; scaleStride=sfStride; headBase=GetSubBlockIdx()*HEADS;
        GM_ADDR base=scratch+uint64_t(GetBlockIdx()/2)*SCRATCH_PER_CORE;
        scoresGm.SetGlobalBuffer((__gm__ float*)base,TOTAL_HEADS*KV_TILE);
        pvGm.SetGlobalBuffer((__gm__ float*)(base+TOTAL_HEADS*KV_TILE*4),TOTAL_HEADS*DIM);

        pipe->InitBuffer(accBuf,HEADS*DIM*4);
        pipe->InitBuffer(arenaBuf,ARENA_BYTES);
        pipe->InitBuffer(probBuf,SCORE_ELEMS*2);
        pipe->InitBuffer(workBuf,WORK_BYTES);
        pipe->InitBuffer(rowBuf,DECODE_ELEMS*2);
        pipe->InitBuffer(decodeOutBuf,2*DECODE_ELEMS*2);
        pipe->InitBuffer(idsBuf,KV_TILE*4);
        pipe->InitBuffer(indexBuf,INDEX_WORDS*4);
        pipe->InitBuffer(maskBuf,64);
        pipe->InitBuffer(lutBuf,768*4);
        pipe->InitBuffer(stateBuf,STATE_BYTES);
        pipe->InitBuffer(ropeBuf,192*4);
        GM_ADDR shared=base+TOTAL_HEADS*(KV_TILE+DIM)*4;
        qPrepared.SetGlobalBuffer((__gm__ bfloat16_t*)shared,TOTAL_HEADS*DIM);
        shared+=TOTAL_HEADS*DIM*2;
        for (int i=0;i<2;++i) kvPrepared[i].SetGlobalBuffer((__gm__ bfloat16_t*)(shared+i*KV_TILE*DIM*2),KV_TILE*DIM);
        shared+=2*KV_TILE*DIM*2;
        pPrepared.SetGlobalBuffer((__gm__ bfloat16_t*)shared,TOTAL_HEADS*KV_TILE);
        InitTables(tables);
    }

    __aicore__ inline void Process() {
        for (uint32_t task=GetBlockIdx()/2;task<taskCount;task+=GetBlockNum()) {
            auto fields=reinterpret_cast<int32_t*>(&cfg);
            for (int i=0;i<12;++i) fields[i]=configsGm.GetValue(uint64_t(task)*12+i);
            for (int i=0;i<ProfileCount;++i) cycles[i]=0;
            auto begin=Tick();
            ProcessQuery();
            cycles[Total]=Tick()-begin;
            if (profiling) {
                auto local=stateBuf.Get<uint64_t>();
                for (int i=0;i<ProfileCount;++i) local.SetValue(i,cycles[i]);
                Sync<HardEvent::S_MTE3>();
                DataCopy(profileGm[(uint64_t(task)*2+GetSubBlockIdx())*ProfileCount],local,ProfileCount);
                Sync<HardEvent::MTE3_S>();
            }
        }
    }

  private:
    template<HardEvent E> __aicore__ inline void Sync() {
        auto id=static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
        SetFlag<E>(id); WaitFlag<E>(id);
    }
    __aicore__ inline uint64_t Tick() { return profiling ? GetSystemCycle() : 0; }
    __aicore__ inline void VDone() { PipeBarrier<PIPE_V>(); }
    __aicore__ inline LocalTensor<float> Work(int slot) { return workBuf.Get<float>()[slot*DIM]; }
    __aicore__ inline float NegInf() { return GetScalarBitcodeValue<uint32_t,float>(0xff800000u); }

    __aicore__ inline void InitTables(GM_ADDR address) {
        GlobalTensor<uint32_t> table;
        table.SetGlobalBuffer((__gm__ uint32_t*)address,TABLE_WORDS);
        DataCopy(lutBuf.Get<uint32_t>(),table,768);
        DataCopy(indexBuf.Get<uint32_t>(),table[768],INDEX_WORDS);
        DataCopy(maskBuf.Get<uint32_t>(),table[768+INDEX_WORDS],16);
        Sync<HardEvent::MTE2_V>();
    }

    __aicore__ inline void ProcessQuery() {
        auto mark=Tick();
        LoadQuery();
        InitOnline();
        cycles[Query]+=Tick()-mark;
        int total=cfg.topk+cfg.extra_topk;
        int tiles=(total+KV_TILE-1)/KV_TILE;
        if (tiles) {
            mark=Tick(); PrepareKv(0,0); cycles[mega_ascend_v3::PrepareKv]+=Tick()-mark;
            StartQk(0);
            if (tiles>1) { mark=Tick(); PrepareKv(1,KV_TILE); cycles[mega_ascend_v3::PrepareKv]+=Tick()-mark; }
        }
        for (int tile=0;tile<tiles;++tile) {
            int slot=tile&1;
            mark=Tick();
            qkMm.WaitIterateAll(); qkMm.End();
            LoadScores(slot);
            cycles[WaitQk]+=Tick()-mark;
            mark=Tick(); OnlineSoftmax(); cycles[Softmax]+=Tick()-mark;
            StartPv(slot);
            // softmax 的 scores/P 已消费；同一 UB 区现在可搬入后两块的 raw KV。
            if (tile+2<tiles) { mark=Tick(); PrefetchKv((tile+2)*KV_TILE); cycles[mega_ascend_v3::PrepareKv]+=Tick()-mark; }
            mark=Tick(); pvMm.WaitIterateAll(); pvMm.End(); cycles[WaitPv]+=Tick()-mark;
            mark=Tick(); AccumulateOutput(); cycles[Accumulate]+=Tick()-mark;

            // 同一 AIV 不同时挂起两种 Matmul；下一块 QK 与后续 KV 解码重叠。
            if (tile+1<tiles) StartQk(slot^1);
            if (tile+2<tiles) {
                mark=Tick(); DecodeKv(slot,(tile+2)*KV_TILE); cycles[mega_ascend_v3::PrepareKv]+=Tick()-mark;
            }
        }
        mark=Tick(); Finalize(); cycles[Finish]+=Tick()-mark;

    }

    __aicore__ inline void LoadQuery() {
        auto rope=ropeBuf.Get<float>();
        DataCopy(rope,ropeGm[uint64_t(cfg.position)*ROPE_DIM],ROPE_DIM);
        Sync<HardEvent::MTE2_V>();
        Gather(rope[64],rope,indexBuf.Get<uint32_t>()[3*DIM+64],0,64);
        Gather(rope[128],rope[32],indexBuf.Get<uint32_t>()[3*DIM+64],0,64);
        VDone();
        Muls(Work(0),rope[128],-1.0f,64); VDone();
        Select(rope[128],maskBuf.Get<uint8_t>(),rope[128],Work(0),SELMODE::VSEL_TENSOR_TENSOR_MODE,64);
        VDone();

        auto row=rowBuf.Get<bfloat16_t>();
        for (int h=0;h<HEADS;++h) {
            // Q 本来就是 [D/16,H,16]：32 个 32 B 块合并成一次带 stride 的搬运。
            DataCopy(row,qGm[uint64_t(cfg.query)*TOTAL_HEADS*DIM+(headBase+h)*16],{32,1,TOTAL_HEADS-1,0});
            Sync<HardEvent::MTE2_V>();
            Cast(Work(0),row,RoundMode::CAST_NONE,DIM); VDone();
            Rotate(Work(0),false);
            Cast(row,Work(0),RoundMode::CAST_RINT,DIM);
            Sync<HardEvent::V_MTE3>();
            DataCopy(qPrepared[(headBase+h)*DIM],row,DIM);
            Sync<HardEvent::MTE3_MTE2>();
        }
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void Rotate(const LocalTensor<float>& row,bool inverse) {
        auto x=row[DIM-ROPE_DIM], paired=Work(1), product=Work(2);
        auto rope=ropeBuf.Get<float>();
        Gather(paired,x,indexBuf.Get<uint32_t>()[3*DIM],0,ROPE_DIM); VDone();
        Mul(paired,paired,rope[128],ROPE_DIM);
        Mul(product,x,rope[64],ROPE_DIM); VDone();
        if (inverse) Sub(x,product,paired,ROPE_DIM);
        else Add(x,product,paired,ROPE_DIM);
        VDone();
    }

    __aicore__ inline void PrepareKv(int slot,int begin) { PrefetchKv(begin); DecodeKv(slot,begin); }
    __aicore__ inline void PrefetchKv(int begin) {
        // arena 即将复用：Vector 清零和 MTE2 搬入都要等此前的 MTE3 读取结束。
        Sync<HardEvent::MTE3_V>();
        Sync<HardEvent::MTE3_MTE2>();
        auto raw=arenaBuf.Get<uint8_t>();
        // 尾行/空行也会参加批量解码，先置零，避免读取未初始化字节。
        Duplicate(raw.ReinterpretCast<uint16_t>(),uint16_t(0),ARENA_BYTES/2);
        Sync<HardEvent::V_MTE2>();
        auto ids=idsBuf.Get<int32_t>();
        int mainRows=cfg.topk-begin; mainRows=mainRows<0?0:(mainRows>KV_TILE?KV_TILE:mainRows);
        int extraBegin=begin+mainRows-cfg.topk;
        int extraRows=cfg.extra_topk-extraBegin;
        extraRows=extraRows<0?0:(extraRows>KV_TILE-mainRows?KV_TILE-mainRows:extraRows);
        DataCopyPadExtParams<int32_t> padIds{false,0,0,0};
        if (mainRows) DataCopyPad(ids,idsGm[uint64_t(cfg.query)*cfg.indices_stride+begin],
            {1,uint32_t(mainRows*4),0,0,0},padIds);
        // extra 起点可能不对齐，分别搬入 work，再用标量读；两个来源不互相覆盖。
        if (extraRows) DataCopyPad(workBuf.Get<int32_t>(),extraIdsGm[uint64_t(cfg.query)*cfg.extra_indices_stride+extraBegin],
            {1,uint32_t(extraRows*4),0,0,0},padIds);
        Sync<HardEvent::MTE2_S>();
        for (int t=0;t<KV_TILE;++t) {
            bool extra=t>=mainRows; int idx=extra?extraBegin+t-mainRows:begin+t;
            int capacity=extra?cfg.extra_topk:cfg.topk, length=extra?cfg.extra_length:cfg.length;
            nextValid[t]=idx<capacity && idx<length;
            if (!nextValid[t] || t<headBase || t>=headBase+HEADS) continue;
            int id=extra?workBuf.Get<int32_t>().GetValue(t-mainRows):ids.GetValue(t);
            int page=extra?cfg.extra_page_size:cfg.page_size;
            int pageStride=extra?cfg.extra_page_stride:cfg.page_stride;
            uint32_t bytes=extra?DIM/2:DIM, scales=extra?DIM/16:DIM/32;
            uint64_t base=uint64_t(id/page)*pageStride;
            auto cache=extra?kv4Gm:kv8Gm;
            int payload=extra?mainRows*DIM+(t-mainRows)*(DIM/2):t*DIM;
            DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
            DataCopyPad(raw[payload],cache[base+uint64_t(id%page)*bytes],{1,bytes,0,0,0},pad);
            DataCopyPad(raw[KV_TILE*DIM+t*32],cache[base+uint64_t(page)*bytes+uint64_t(id%page)*scales],
                {1,scales,0,0,0},pad);
        }
    }

    __aicore__ inline void ByteIndices(const LocalTensor<uint8_t>& src,int count) {
        auto offsets=workBuf.Get<int32_t>()[2*DECODE_ELEMS];
        Cast(rowBuf.Get<half>(),src,RoundMode::CAST_NONE,count); VDone();
        Cast(offsets,rowBuf.Get<half>(),RoundMode::CAST_RINT,count); VDone();
        ShiftLeft(offsets.ReinterpretCast<uint32_t>(),offsets.ReinterpretCast<uint32_t>(),uint32_t(2),count); VDone();
    }

    __aicore__ inline void DecodeKv(int slot,int begin) {
        Sync<HardEvent::MTE2_V>();
        auto raw=arenaBuf.Get<uint8_t>();
        auto lut=lutBuf.Get<float>();
        auto values=workBuf.Get<float>(), scales=values[DECODE_ELEMS];
        auto offsets=values[2*DECODE_ELEMS].ReinterpretCast<uint32_t>();

        int mainRows=cfg.topk-begin; mainRows=mainRows<0?0:(mainRows>KV_TILE?KV_TILE:mainRows);
        for (int t=0;t<KV_TILE;++t) valid[slot][t]=nextValid[t];
        int batch=0;
        for (int t=headBase;t<headBase+HEADS;++batch) {
            bool fp4=t>=mainRows;
            int limit=headBase+HEADS;
            int rows=fp4?limit-t:(mainRows<limit?mainRows:limit)-t;
            rows=rows>DECODE_ROWS?DECODE_ROWS:rows;
            int elems=rows*DIM, payload=fp4?mainRows*DIM+(t-mainRows)*(DIM/2):t*DIM;
            ByteIndices(raw[payload],fp4?elems/2:elems);
            if (fp4) {
                Gather(rowBuf.Get<uint32_t>(),lutBuf.Get<uint32_t>()[512],offsets,0,elems/2); VDone();
                Cast(values,rowBuf.Get<bfloat16_t>(),RoundMode::CAST_NONE,elems); VDone();
            } else {
                Gather(values,lut,offsets,0,elems); VDone();
            }
            ByteIndices(raw[KV_TILE*DIM+t*32],rows*32);
            Gather(scales,lut[fp4?0:256],offsets,0,rows*32); VDone();
            Gather(values[2*DECODE_ELEMS],scales,
                indexBuf.Get<uint32_t>()[3*DIM+128+(fp4?DECODE_ELEMS:0)],0,elems); VDone();
            Mul(values,values,values[2*DECODE_ELEMS],elems); VDone();
            // 两块 BF16 缓冲交替写；只在即将复用前等待上一轮 MTE3。
            auto output=decodeOutBuf.Get<bfloat16_t>()[(batch&1)*DECODE_ELEMS];
            Sync<HardEvent::MTE3_V>();
            Cast(output,values,RoundMode::CAST_RINT,elems);
            Sync<HardEvent::V_MTE3>();
            DataCopy(kvPrepared[slot][t*DIM],output,elems);
            t+=rows;
        }
        Sync<HardEvent::MTE3_S>();
    }

    __aicore__ inline void InitOnline() {
        // 用最小有限值初始化，空块的最大值差不会出现 -inf-(-inf)。
        Duplicate(stateBuf.Get<float>(),-3.402823466e38f,HEADS);
        Duplicate(stateBuf.Get<float>()[64],0.0f,HEADS);
        Duplicate(accBuf.Get<float>(),0.0f,HEADS*DIM);
        Sync<HardEvent::V_S>();
    }
    __aicore__ inline void StartQk(int slot) {
        auto mark=Tick();
        qkMm.SetTensorA(qPrepared); qkMm.SetTensorB(kvPrepared[slot],true);
        qkMm.IterateAll<false>(scoresGm,0,false,true);
        cycles[WaitQk]+=Tick()-mark;
    }
    __aicore__ inline void LoadScores(int slot) {
        auto scores=arenaBuf.Get<float>(), mask=Work(0);
        for (int t=0;t<KV_TILE;++t) mask.SetValue(t,valid[slot][t]?0.0f:NegInf());
        Sync<HardEvent::S_V>();
        DataCopy(scores,scoresGm[headBase*KV_TILE],SCORE_ELEMS); Sync<HardEvent::MTE2_V>();
        // 下一块 QK 会覆盖同一 GM 输出，标量发射前必须确认这次读取已完成。
        Sync<HardEvent::MTE2_S>();
        Muls(scores,scores,SCORE_SCALE,SCORE_ELEMS); VDone();
        for (int h=0;h<HEADS;++h) Add(scores[h*KV_TILE],scores[h*KV_TILE],mask,KV_TILE);
        VDone();
    }
    __aicore__ inline void OnlineSoftmax() {
        auto scores=arenaBuf.Get<float>();
        auto prob=probBuf.Get<bfloat16_t>();
        auto m=stateBuf.Get<float>(), sum=m[64], alpha=m[128], tileMax=m[192], tileSum=m[256];
        auto delta=Work(0), safeMax=Work(1), broadcast=Work(2);
        auto update=rowBuf.Get<uint8_t>();
        WholeReduceMax(tileMax,scores,KV_TILE,HEADS,1,1,8,ReduceOrder::ORDER_ONLY_VALUE); VDone();
        Adds(alpha,m,0.0f,HEADS);
        Sub(delta,tileMax,m,HEADS); VDone();
        // Compares 的 calCount 版本要求 64 个 float；用显式 mask 处理 32 head。
        Compares(update,delta,RESCALE_DELTA,CMPMODE::GT,uint64_t(HEADS),1,{1,1,8,8}); VDone();
        Select(m,update,tileMax,m,SELMODE::VSEL_TENSOR_TENSOR_MODE,HEADS); VDone();
        // 有限下界配合被 mask 成 -inf 的分数，让空行的 exp 和行和均为 0。
        Maxs(safeMax,m,-3.402823466e38f,HEADS); VDone();
        Sub(alpha,alpha,safeMax,HEADS);
        Brcb(broadcast,safeMax,HEADS/8,{1,8}); VDone();
        // 每个 head 广播成一个 32B 块，src1 的块步长为 0，覆盖整行 64 个分数。
        Sub(scores,scores,broadcast,64,HEADS,{1,1,0,8,8,1}); VDone();
        Exp(scores,scores,SCORE_ELEMS); Exp(alpha,alpha,HEADS); VDone();
        WholeReduceSum(tileSum,scores,KV_TILE,HEADS,1,1,8);
        Cast(prob,scores,RoundMode::CAST_RINT,SCORE_ELEMS); VDone();
        Mul(sum,sum,alpha,HEADS); VDone(); Add(sum,sum,tileSum,HEADS); VDone();

        Sync<HardEvent::V_MTE3>();
        DataCopy(pPrepared[headBase*KV_TILE],prob,SCORE_ELEMS);
        Sync<HardEvent::MTE3_S>();
    }
    __aicore__ inline void StartPv(int slot) {
        auto mark=Tick();
        pvMm.SetTensorA(pPrepared); pvMm.SetTensorB(kvPrepared[slot],false);
        pvMm.IterateAll<false>(pvGm,0,false,true);
        cycles[WaitPv]+=Tick()-mark;
    }
    __aicore__ inline void AccumulateOutput() {
        auto acc=accBuf.Get<float>(), pv=workBuf.Get<float>();
        auto broadcast=workBuf.Get<float>()[2*DECODE_ELEMS];
        auto alpha=stateBuf.Get<float>()[128];
        Brcb(broadcast,alpha,HEADS/8,{1,8}); VDone();
        for (int h=0;h<HEADS;h+=8) {
            DataCopy(pv,pvGm[(headBase+h)*DIM],8*DIM); Sync<HardEvent::MTE2_V>();
            for (int j=0;j<8;++j)
                Mul(acc[(h+j)*DIM],acc[(h+j)*DIM],broadcast[(h+j)*8],64,DIM/64,{1,1,0,8,8,0});
            VDone(); Add(acc[h*DIM],acc[h*DIM],pv,8*DIM);
            Sync<HardEvent::V_MTE2>();
        }
        Sync<HardEvent::V_S>();
    }

    __aicore__ inline void Finalize() {
        auto m=stateBuf.Get<float>(), sum=m[64], lse=m[192], gate=m[256], factor=m[128];
        float inf=GetScalarBitcodeValue<uint32_t,float>(0x7f800000u);
        for (int h=0;h<HEADS;++h) lse.SetValue(h,sum.GetValue(h)==0?1.0f:sum.GetValue(h));
        Sync<HardEvent::S_V>(); Ln(lse,lse,HEADS); Sync<HardEvent::V_S>();
        for (int h=0;h<HEADS;++h) {
            bool empty=sum.GetValue(h)==0;
            float l=empty?inf:m.GetValue(h)+lse.GetValue(h);
            lse.SetValue(h,l);
            float delta=empty?0.0f:(hasSink?sinkGm.GetValue(headBase+h):NegInf())-l;
            factor.SetValue(h,delta); gate.SetValue(h,delta>0?-delta:delta);
        }
        Sync<HardEvent::S_V>(); Exp(gate,gate,HEADS); Sync<HardEvent::V_S>();
        for (int h=0;h<HEADS;++h) {
            float s=sum.GetValue(h), t=gate.GetValue(h), delta=factor.GetValue(h);
            float g=delta>=0?t/(1+t):1/(1+t);
            factor.SetValue(h,s==0?0.0f:g/s);
        }
        Sync<HardEvent::S_MTE3>(); DataCopy(lseGm[uint64_t(cfg.query)*TOTAL_HEADS+headBase],lse,HEADS);
        Sync<HardEvent::MTE3_S>();
        for (int h=0;h<HEADS;++h) {
            auto row=accBuf.Get<float>()[h*DIM];
            float f=factor.GetValue(h);
            if (f==0) Duplicate(row,0.0f,DIM); else Muls(row,row,f,DIM);
            VDone(); Rotate(row,true);
        }
        for (int h=0;h<HEADS;h+=DECODE_ROWS) QuantizeRows(accBuf.Get<float>()[h*DIM],h);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void QuantizeRows(const LocalTensor<float>& row,int h) {
        auto abs=workBuf.Get<float>(), exp=abs[DECODE_ELEMS], inv=abs[2*DECODE_ELEMS];
        auto temp=arenaBuf.Get<float>(), code=temp[DECODE_ELEMS];
        auto groupMax=temp[2*DECODE_ELEMS], invScales=groupMax[64];
        auto words=invScales[128].ReinterpretCast<uint32_t>();
        Abs(abs,row,DECODE_ELEMS); VDone();
        WholeReduceMax(groupMax,abs,32,DECODE_ROWS*16,1,1,4,ReduceOrder::ORDER_ONLY_VALUE); VDone();
        Maxs(groupMax,groupMax,1e-4f,64); VDone();
        Muls(groupMax,groupMax,1.0f/448.0f,64); VDone();
        // bits-1 把恰好为 2 的幂的边界归入较小指数；e=E-126，scale 字节=E+1。
        Adds(code.ReinterpretCast<int32_t>(),groupMax.ReinterpretCast<int32_t>(),int32_t(-1),64); VDone();
        ShiftRight(code.ReinterpretCast<uint32_t>(),code.ReinterpretCast<uint32_t>(),uint32_t(23),64); VDone();
        Cast(exp,code.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,64); VDone();
        Adds(groupMax,exp,1.0f,64); VDone();
        Cast(code.ReinterpretCast<int32_t>(),groupMax,RoundMode::CAST_RINT,64); VDone();
        // 1/scale 的 FP32 指数位为 127-e=253-E，尾数为 0。
        Muls(invScales,exp,-1.0f,64); VDone(); Adds(invScales,invScales,253.0f,64); VDone();
        Cast(invScales.ReinterpretCast<int32_t>(),invScales,RoundMode::CAST_RINT,64); VDone();
        ShiftLeft(invScales.ReinterpretCast<uint32_t>(),invScales.ReinterpretCast<uint32_t>(),uint32_t(23),64);
        Sync<HardEvent::V_S>();
        auto scaleCodes=code.ReinterpretCast<uint32_t>();
        for (int g=0;g<16;++g)
            words.SetValue(g*8,scaleCodes.GetValue(g)|(scaleCodes.GetValue(16+g)<<8)|
                (scaleCodes.GetValue(32+g)<<16)|(scaleCodes.GetValue(48+g)<<24));
        for (int j=0;j<DECODE_ROWS;++j)
            Gather(inv[j*DIM],invScales[j*16],indexBuf.Get<uint32_t>()[DIM],0,DIM);
        VDone();
        Mul(abs,abs,inv,DECODE_ELEMS); VDone(); Mins(abs,abs,448.0f,DECODE_ELEMS); VDone();
        // E=max(FP32 指数,121)；量化步长为 2^(E-130)，非规格化段固定为 2^-9。
        ShiftRight(temp.ReinterpretCast<uint32_t>(),abs.ReinterpretCast<uint32_t>(),uint32_t(23),DECODE_ELEMS); VDone();
        Cast(exp,temp.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DECODE_ELEMS); VDone();
        Maxs(exp,exp,121.0f,DECODE_ELEMS); VDone();
        Muls(inv,exp,-1.0f,DECODE_ELEMS); VDone(); Adds(inv,inv,257.0f,DECODE_ELEMS); VDone();
        Cast(temp.ReinterpretCast<int32_t>(),inv,RoundMode::CAST_RINT,DECODE_ELEMS); VDone();
        ShiftLeft(temp.ReinterpretCast<uint32_t>(),temp.ReinterpretCast<uint32_t>(),uint32_t(23),DECODE_ELEMS); VDone();
        Mul(inv,abs,temp,DECODE_ELEMS); VDone();
        Cast(code.ReinterpretCast<int32_t>(),inv,RoundMode::CAST_RINT,DECODE_ELEMS); VDone();
        Cast(code,code.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DECODE_ELEMS); VDone();
        Adds(exp,exp,-121.0f,DECODE_ELEMS); VDone(); Muls(exp,exp,8.0f,DECODE_ELEMS); VDone();
        Add(code,code,exp,DECODE_ELEMS);
        ShiftRight(temp.ReinterpretCast<uint32_t>(),row.ReinterpretCast<uint32_t>(),uint32_t(31),DECODE_ELEMS); VDone();
        Cast(temp,temp.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DECODE_ELEMS); VDone();
        Muls(temp,temp,128.0f,DECODE_ELEMS); VDone(); Add(code,code,temp,DECODE_ELEMS); VDone();
        // 码字 0..254 可经 half 精确转成 uint8；符号位也保留负零。
        Cast(rowBuf.Get<half>(),code,RoundMode::CAST_RINT,DECODE_ELEMS); VDone();
        auto packed=abs.ReinterpretCast<uint8_t>();
        Cast(packed,rowBuf.Get<half>(),RoundMode::CAST_RINT,DECODE_ELEMS);
        Sync<HardEvent::V_MTE3>();
        for (int j=0;j<DECODE_ROWS;++j) {
            int globalHead=headBase+h+j;
            uint64_t base=uint64_t(cfg.query)*TOTAL_HEADS*DIM+(globalHead/8)*4096+(globalHead%8)*32;
            DataCopy(fp8Gm[base],packed[j*DIM],{16,1,0,7});
        }
        Sync<HardEvent::S_MTE3>();
        for (int g=0;g<16;++g) {
            int globalHead=headBase+h;
            int word=(globalHead/8)*32+g*2+(globalHead%8)/4;
            DataCopyPad(scalesGm[uint64_t(word)*scaleStride+cfg.query],words[g*8],{1,4,0,0,0});
        }
        Sync<HardEvent::MTE3_V>(); Sync<HardEvent::MTE3_S>();
    }

    QueryConfig cfg;
    uint32_t taskCount; int32_t scaleStride,headBase; bool hasSink,profiling;
    bool nextValid[KV_TILE],valid[2][KV_TILE];
    uint64_t cycles[ProfileCount];
    GlobalTensor<bfloat16_t> qGm;
    GlobalTensor<uint8_t> kv8Gm,kv4Gm,fp8Gm;
    GlobalTensor<int32_t> idsGm,extraIdsGm,configsGm;
    GlobalTensor<float> ropeGm,sinkGm,lseGm,scoresGm,pvGm;
    GlobalTensor<uint32_t> scalesGm;
    GlobalTensor<uint64_t> profileGm;
    TBuf<TPosition::VECCALC> accBuf,workBuf,indexBuf,maskBuf,lutBuf,stateBuf,ropeBuf;
    TBuf<TPosition::VECOUT> arenaBuf,rowBuf,decodeOutBuf,probBuf;
    TBuf<TPosition::VECCALC> idsBuf;
    GlobalTensor<bfloat16_t> qPrepared,pPrepared,kvPrepared[2];
};

extern "C" __global__ __aicore__ void mega_attn_v3(
    GM_ADDR q,GM_ADDR kv8,GM_ADDR kv4,GM_ADDR ids,GM_ADDR extraIds,
    GM_ADDR rope,GM_ADDR sink,GM_ADDR configs,uint32_t count,
    GM_ADDR fp8,GM_ADDR scales,GM_ADDR lse,int32_t sfStride,
    GM_ADDR scratch,GM_ADDR profiles,GM_ADDR tables,GM_ADDR qkTilingGm,GM_ADDR pvTilingGm,GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe; KernelMegaAttentionV3 op; TCubeTiling qk,pv;
    auto a=(__gm__ uint32_t*)qkTilingGm, b=(__gm__ uint32_t*)pvTilingGm;
    for (int i=0;i<sizeof(TCubeTiling)/4;++i) {
        ((uint32_t*)&qk)[i]=a[i]; ((uint32_t*)&pv)[i]=b[i];
    }
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),op.qkMm,&qk,op.pvMm,&pv);
    op.Init(q,kv8,kv4,ids,extraIds,rope,sink,configs,count,fp8,scales,lse,sfStride,scratch,profiles,tables,&pipe);
    op.Process();
}

extern "C" void launch_mega_attn_v3(
    const Input &in,const Output &out,const QueryConfig *configs,uint32_t count,uint32_t blocks,
    void *scratch,void *qkTiling,void *pvTiling,void *workspace,void *tables,void *stream,Profile *profiles
) {
    if (!count) return;
    mega_attn_v3<<<blocks,nullptr,stream>>>(
        (uint8_t*)in.q,(uint8_t*)in.kv_fp8,(uint8_t*)in.kv_fp4,(uint8_t*)in.indices,
        (uint8_t*)in.extra_indices,(uint8_t*)in.cos_sin,(uint8_t*)in.sink,(uint8_t*)configs,count,
        out.fp8,(uint8_t*)out.scales,(uint8_t*)out.lse,out.sf_stride,(uint8_t*)scratch,
        (uint8_t*)profiles,(uint8_t*)tables,(uint8_t*)qkTiling,(uint8_t*)pvTiling,(uint8_t*)workspace);
}
