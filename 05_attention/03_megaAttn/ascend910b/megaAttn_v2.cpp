#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "megaAttn_v2.h"

using namespace AscendC;
using namespace mega_ascend_v2;

// VECOUT 标记 TSCM 的生产者在 AIV；AIC 直接读取 L1 中的 NZ 矩阵。
using AType=MatmulType<TPosition::TSCM,CubeFormat::NZ,bfloat16_t,false,LayoutMode::NONE,false,TPosition::VECOUT>;
using KType=MatmulType<TPosition::TSCM,CubeFormat::NZ,bfloat16_t,true,LayoutMode::NONE,false,TPosition::VECOUT>;
using CType=MatmulType<TPosition::GM,CubeFormat::ND,float>;

// 一组 Cube/Vector 处理一个完整 query；两个 L1 KV 槽交替供 QK 和 PV 使用。
class KernelMegaAttentionV2 {
  public:
    Matmul<AType,KType,CType> qkMm;
    Matmul<AType,AType,CType> pvMm;

    __aicore__ inline void Init(
        GM_ADDR q,GM_ADDR kv8,GM_ADDR kv4,GM_ADDR ids,GM_ADDR extraIds,
        GM_ADDR rope,GM_ADDR sink,GM_ADDR configs,uint32_t count,
        GM_ADDR fp8,GM_ADDR scales,GM_ADDR lse,int32_t sfStride,
        GM_ADDR scratch,GM_ADDR profiles,TPipe *pipe
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
        taskCount=count; scaleStride=sfStride;
        GM_ADDR base=scratch+uint64_t(GetBlockIdx())*SCRATCH_PER_CORE;
        scoresGm.SetGlobalBuffer((__gm__ float*)base,SCORE_ELEMS);
        pvGm.SetGlobalBuffer((__gm__ float*)(base+SCORE_ELEMS*4),HEADS*DIM);

        pipe->InitBuffer(accBuf,HEADS*DIM*4);
        pipe->InitBuffer(arenaBuf,ARENA_BYTES);
        pipe->InitBuffer(workBuf,5*DIM*4);
        pipe->InitBuffer(rowBuf,DIM*2);
        pipe->InitBuffer(indexBuf,3*DIM*4+128*4);
        pipe->InitBuffer(maskBuf,64);
        pipe->InitBuffer(lutBuf,528*4);
        pipe->InitBuffer(stateBuf,2048);
        pipe->InitBuffer(ropeBuf,192*4);
        pipe->InitBuffer(qQueue,1,HEADS*DIM*2);
        pipe->InitBuffer(kvQueue,2,KV_TILE*DIM*2);
        pipe->InitBuffer(pQueue,1,HEADS*KV_TILE*2);
        InitTables();
    }

    __aicore__ inline void Process() {
        for (uint32_t task=GetBlockIdx();task<taskCount;task+=GetBlockNum()) {
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
                DataCopy(profileGm[uint64_t(task)*ProfileCount],local,ProfileCount);
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

    __aicore__ inline void InitTables() {
        auto lut=lutBuf.Get<uint32_t>();
        for (int i=0;i<256;++i) {
            lut.SetValue(i,E4M3Bits(uint8_t(i)));
            lut.SetValue(256+i,E8M0Bits(uint8_t(i)));
        }
        constexpr float fp4[8]={0,.5f,1,1.5f,2,3,4,6};
        auto values=lutBuf.Get<float>();
        for (int i=0;i<16;++i) values.SetValue(512+i,(i&8)?-fp4[i&7]:fp4[i&7]);
        auto idx=indexBuf.Get<uint32_t>();
        for (int d=0;d<DIM;++d) {
            idx.SetValue(d,(d/2)*4);           // FP4 一个字节供两个维度使用
            idx.SetValue(DIM+d,(d/32)*4);      // FP8 scale 广播
            idx.SetValue(2*DIM+d,(d/16)*4);    // FP4 scale 广播
        }
        for (int d=0;d<ROPE_DIM;++d) {
            idx.SetValue(3*DIM+d,(d^1)*4);     // RoPE 相邻偶/奇维配对
            idx.SetValue(3*DIM+64+d,(d/2)*4);  // 一个 cos/sin 供一对维度使用
        }
        auto mask=maskBuf.Get<uint32_t>();
        for (int i=0;i<16;++i) mask.SetValue(i,0xaaaaaaaau);
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void ProcessQuery() {
        auto mark=Tick();
        LoadQuery();
        InitOnline();
        cycles[Query]+=Tick()-mark;
        int total=cfg.topk+cfg.extra_topk;
        int tiles=(total+KV_TILE-1)/KV_TILE;
        if (tiles) {
            mark=Tick(); PrepareKv(0,0); cycles[mega_ascend_v2::PrepareKv]+=Tick()-mark;
            StartQk(0);
            if (tiles>1) { mark=Tick(); PrepareKv(1,KV_TILE); cycles[mega_ascend_v2::PrepareKv]+=Tick()-mark; }
        }
        for (int tile=0;tile<tiles;++tile) {
            int slot=tile&1;
            mark=Tick();
            qkMm.WaitIterateAll(); qkMm.End();
            LoadScores(slot);
            cycles[WaitQk]+=Tick()-mark;
            // 当前分数已在 UB，Cube 可以覆盖 GM 分数，计算下一块 QK。
            if (tile+1<tiles) StartQk(slot^1);
            mark=Tick(); OnlineSoftmax(); cycles[Softmax]+=Tick()-mark;
            StartPv(slot);
            // softmax 的 scores/P 已消费；同一 UB 区现在可搬入后两块的 raw KV。
            if (tile+2<tiles) { mark=Tick(); PrefetchKv((tile+2)*KV_TILE); cycles[mega_ascend_v2::PrepareKv]+=Tick()-mark; }
            mark=Tick(); pvMm.WaitIterateAll(); pvMm.End(); cycles[WaitPv]+=Tick()-mark;
            mark=Tick(); AccumulateOutput(); cycles[Accumulate]+=Tick()-mark;
            kvQueue.FreeTensor(kvL1[slot]); pQueue.FreeTensor(pL1);
            if (tile+2<tiles) {
                mark=Tick(); DecodeKv(slot,(tile+2)*KV_TILE); cycles[mega_ascend_v2::PrepareKv]+=Tick()-mark;
            }
        }
        mark=Tick(); Finalize(); cycles[Finish]+=Tick()-mark;
        qQueue.FreeTensor(qL1);
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
        qL1=qQueue.AllocTensor<bfloat16_t>();
        auto row=rowBuf.Get<bfloat16_t>();
        for (int h=0;h<HEADS;++h) {
            // Q 本来就是 [D/16,H,16]：32 个 32 B 块合并成一次带 stride 的搬运。
            DataCopy(row,qGm[uint64_t(cfg.query)*HEADS*DIM+h*16],{32,1,HEADS-1,0});
            Sync<HardEvent::MTE2_V>();
            Cast(Work(0),row,RoundMode::CAST_NONE,DIM); VDone();
            Rotate(Work(0),false);
            Cast(row,Work(0),RoundMode::CAST_RINT,DIM);
            Sync<HardEvent::V_MTE3>();
            StoreNzRow(qL1,row,h,HEADS,DIM);
            Sync<HardEvent::MTE3_MTE2>();
        }
        qQueue.EnQue(qL1); qL1=qQueue.DeQue<bfloat16_t>();
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

    __aicore__ inline void StoreNzRow(const LocalTensor<bfloat16_t>& dst,
        const LocalTensor<bfloat16_t>& src,int row,int rows,int cols) {
        // NZ[(d/16),row,d%16]：每块 16 个 BF16，相邻列块跨 rows 个 32 B 块。
        DataCopy(dst[row*16],src,{uint16_t(cols/16),1,0,uint16_t(rows-1)});
    }

    __aicore__ inline void PrepareKv(int slot,int begin) { PrefetchKv(begin); DecodeKv(slot,begin); }
    __aicore__ inline void PrefetchKv(int begin) {
        // arena 之前可能由 MTE3 读取 P；必须读完才能由 MTE2 覆写。
        Sync<HardEvent::MTE3_MTE2>();
        auto raw=arenaBuf.Get<uint8_t>();
        for (int t=0;t<KV_TILE;++t) {
            int pos=begin+t; bool extra=pos>=cfg.topk;
            int idx=extra?pos-cfg.topk:pos;
            int capacity=extra?cfg.extra_topk:cfg.topk;
            int length=extra?cfg.extra_length:cfg.length;
            nextValid[t]=idx<capacity && idx<length;
            if (!nextValid[t]) continue;
            auto ids=extra?extraIdsGm:idsGm;
            int stride=extra?cfg.extra_indices_stride:cfg.indices_stride;
            int id=ids.GetValue(uint64_t(cfg.query)*stride+idx);
            int page=extra?cfg.extra_page_size:cfg.page_size;
            int pageStride=extra?cfg.extra_page_stride:cfg.page_stride;
            uint32_t bytes=extra?DIM/2:DIM, scales=extra?DIM/16:DIM/32;
            uint64_t base=uint64_t(id/page)*pageStride;
            auto cache=extra?kv4Gm:kv8Gm;
            DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
            DataCopyPad(raw[t*RAW_ROW_BYTES],cache[base+uint64_t(id%page)*bytes],{1,bytes,0,0,0},pad);
            DataCopyPad(raw[t*RAW_ROW_BYTES+DIM],cache[base+uint64_t(page)*bytes+uint64_t(id%page)*scales],{1,scales,0,0,0},pad);
        }
    }

    __aicore__ inline void ByteIndices(const LocalTensor<uint8_t>& src,int count) {
        // A2 支持 uint8→half→int32；0..255 可精确表示，再乘 4 得到查表字节偏移。
        auto halfRow=rowBuf.Get<half>();
        auto offsets=Work(4).ReinterpretCast<int32_t>();
        Cast(halfRow,src,RoundMode::CAST_NONE,count); VDone();
        Cast(offsets,halfRow,RoundMode::CAST_RINT,count); VDone();
        ShiftLeft(offsets.ReinterpretCast<uint32_t>(),offsets.ReinterpretCast<uint32_t>(),uint32_t(2),count); VDone();
    }

    __aicore__ inline void DecodeKv(int slot,int begin) {
        Sync<HardEvent::MTE2_V>();
        auto raw=arenaBuf.Get<uint8_t>();
        auto lut=lutBuf.Get<float>();
        auto offsets=Work(4).ReinterpretCast<uint32_t>();
        auto values=Work(0), scales=Work(3);
        auto row=rowBuf.Get<bfloat16_t>();
        kvL1[slot]=kvQueue.AllocTensor<bfloat16_t>();
        for (int t=0;t<KV_TILE;++t) {
            valid[slot][t]=nextValid[t];
            if (!nextValid[t]) { Duplicate(row,GetScalarBitcodeValue<uint16_t,bfloat16_t>(0),DIM); }
            else {
                bool fp4=begin+t>=cfg.topk;
                if (fp4) {
                    // 先把一个字节扩成两个相邻 float，再用整除 16 拆出高低半字节。
                    Cast(rowBuf.Get<half>(),raw[t*RAW_ROW_BYTES],RoundMode::CAST_NONE,DIM/2); VDone();
                    Cast(Work(1),rowBuf.Get<half>(),RoundMode::CAST_NONE,DIM/2); VDone();
                    Gather(values,Work(1),indexBuf.Get<uint32_t>(),0,DIM); VDone();
                    Muls(Work(1),values,1.0f/16,DIM); VDone();
                    Cast(Work(4).ReinterpretCast<int32_t>(),Work(1),RoundMode::CAST_FLOOR,DIM); VDone();
                    Cast(Work(1),Work(4).ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DIM); VDone();
                    Muls(Work(2),Work(1),16.0f,DIM); VDone();
                    Sub(Work(2),values,Work(2),DIM); VDone();
                    Select(values,maskBuf.Get<uint8_t>(),Work(1),Work(2),SELMODE::VSEL_TENSOR_TENSOR_MODE,DIM); VDone();
                    Cast(Work(4).ReinterpretCast<int32_t>(),values,RoundMode::CAST_RINT,DIM); VDone();
                    ShiftLeft(offsets,offsets,uint32_t(2),DIM); VDone();
                    Gather(values,lut[512],offsets,0,DIM); VDone();
                } else {
                    ByteIndices(raw[t*RAW_ROW_BYTES],DIM);
                    Gather(values,lut,offsets,0,DIM); VDone();
                }
                int groups=fp4?32:16;
                ByteIndices(raw[t*RAW_ROW_BYTES+DIM],groups);
                Gather(scales,lut[fp4?0:256],offsets,0,groups); VDone();
                Gather(Work(1),scales,indexBuf.Get<uint32_t>()[fp4?2*DIM:DIM],0,DIM); VDone();
                Mul(values,values,Work(1),DIM); VDone();
                Cast(row,values,RoundMode::CAST_RINT,DIM);
            }
            Sync<HardEvent::V_MTE3>();
            StoreNzRow(kvL1[slot],row,t,KV_TILE,DIM);
            // 下一行的 Cast 会复用 rowBuf，必须等 MTE3 读完。
            Sync<HardEvent::MTE3_V>();
        }
        kvQueue.EnQue(kvL1[slot]); kvL1[slot]=kvQueue.DeQue<bfloat16_t>();
    }

    __aicore__ inline void InitOnline() {
        Duplicate(stateBuf.Get<float>(),NegInf(),HEADS);
        Duplicate(stateBuf.Get<float>()[64],0.0f,HEADS);
        Duplicate(accBuf.Get<float>(),0.0f,HEADS*DIM);
        Sync<HardEvent::V_S>();
    }
    __aicore__ inline void StartQk(int slot) {
        qkMm.SetTensorA(qL1); qkMm.SetTensorB(kvL1[slot],true);
        qkMm.IterateAll<false>(scoresGm,0,false,true);
    }
    __aicore__ inline void LoadScores(int slot) {
        auto scores=arenaBuf.Get<float>(), mask=Work(0);
        for (int t=0;t<KV_TILE;++t) mask.SetValue(t,valid[slot][t]?0.0f:NegInf());
        Sync<HardEvent::S_V>();
        DataCopy(scores,scoresGm,SCORE_ELEMS); Sync<HardEvent::MTE2_V>();
        // 下一块 QK 会覆盖同一 GM 输出，标量发射前必须确认这次读取已完成。
        Sync<HardEvent::MTE2_S>();
        Muls(scores,scores,SCORE_SCALE,SCORE_ELEMS); VDone();
        for (int h=0;h<HEADS;++h) Add(scores[h*KV_TILE],scores[h*KV_TILE],mask,KV_TILE);
        VDone();
    }
    __aicore__ inline void OnlineSoftmax() {
        auto scores=arenaBuf.Get<float>();
        auto prob=arenaBuf.Get<bfloat16_t>()[SCORE_ELEMS*2];
        auto m=stateBuf.Get<float>(), sum=m[64], alpha=m[128], tileMax=m[192], tileSum=m[256];
        WholeReduceMax(tileMax,scores,KV_TILE,HEADS,1,1,8,ReduceOrder::ORDER_ONLY_VALUE);
        Sync<HardEvent::V_S>();
        for (int h=0;h<HEADS;++h) {
            float old=m.GetValue(h), current=tileMax.GetValue(h);
            // 延迟 rescale：只有基准落后 ln(64) 才更新，减少整行输出的乘法。
            float next=(old==NegInf() || current-old>RESCALE_DELTA)?current:old;
            bool empty=next==NegInf();
            m.SetValue(h,next); alpha.SetValue(h,empty?0.0f:old-next);
            if (empty) Duplicate(scores[h*KV_TILE],NegInf(),KV_TILE);
            else Adds(scores[h*KV_TILE],scores[h*KV_TILE],-next,KV_TILE);
        }
        Sync<HardEvent::S_V>(); VDone();
        Exp(scores,scores,SCORE_ELEMS); Exp(alpha,alpha,HEADS); VDone();
        WholeReduceSum(tileSum,scores,KV_TILE,HEADS,1,1,8);
        Cast(prob,scores,RoundMode::CAST_RINT,SCORE_ELEMS);
        Sync<HardEvent::V_S>();
        for (int h=0;h<HEADS;++h) sum.SetValue(h,alpha.GetValue(h)*sum.GetValue(h)+tileSum.GetValue(h));
        pL1=pQueue.AllocTensor<bfloat16_t>();
        Sync<HardEvent::V_MTE3>();
        for (int h=0;h<HEADS;++h) StoreNzRow(pL1,prob[h*KV_TILE],h,HEADS,KV_TILE);
        pQueue.EnQue(pL1); pL1=pQueue.DeQue<bfloat16_t>();
    }
    __aicore__ inline void StartPv(int slot) {
        pvMm.SetTensorA(pL1); pvMm.SetTensorB(kvL1[slot],false);
        pvMm.IterateAll<false>(pvGm,0,false,true);
    }
    __aicore__ inline void AccumulateOutput() {
        auto acc=accBuf.Get<float>(), pv=Work(0), alpha=stateBuf.Get<float>()[128];
        // 4 个 head=8 KiB；复用 work，避免为完整 128 KiB PV 再申请一份 UB。
        for (int h=0;h<HEADS;h+=4) {
            DataCopy(pv,pvGm[h*DIM],4*DIM); Sync<HardEvent::MTE2_V>();
            for (int j=0;j<4;++j) {
                float a=alpha.GetValue(h+j);
                if (a!=1.0f) Muls(acc[(h+j)*DIM],acc[(h+j)*DIM],a,DIM);
            }
            VDone(); Add(acc[h*DIM],acc[h*DIM],pv,4*DIM);
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
            float delta=empty?0.0f:(hasSink?sinkGm.GetValue(h):NegInf())-l;
            factor.SetValue(h,delta); gate.SetValue(h,delta>0?-delta:delta);
        }
        Sync<HardEvent::S_V>(); Exp(gate,gate,HEADS); Sync<HardEvent::V_S>();
        for (int h=0;h<HEADS;++h) {
            float s=sum.GetValue(h), t=gate.GetValue(h), delta=factor.GetValue(h);
            float g=delta>=0?t/(1+t):1/(1+t);
            factor.SetValue(h,s==0?0.0f:g/s);
        }
        Sync<HardEvent::S_MTE3>(); DataCopy(lseGm[uint64_t(cfg.query)*HEADS],lse,HEADS);
        Sync<HardEvent::MTE3_S>();
        for (int h=0;h<HEADS;++h) {
            auto row=accBuf.Get<float>()[h*DIM];
            float f=factor.GetValue(h);
            if (f==0) Duplicate(row,0.0f,DIM); else Muls(row,row,f,DIM);
            VDone(); Rotate(row,true); QuantizeRow(row,h);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void QuantizeRow(const LocalTensor<float>& row,int h) {
        auto abs=Work(0), exp=Work(1), inv=Work(2), temp=Work(3), code=Work(4);
        auto groupMax=stateBuf.Get<float>()[320], invScales=stateBuf.Get<float>()[352];
        auto words=stateBuf.Get<uint32_t>()[384];
        if (h%4==0) for (int g=0;g<16;++g) words.SetValue(g*8,0);
        Abs(abs,row,DIM); VDone();
        WholeReduceMax(groupMax,abs,32,16,1,1,4,ReduceOrder::ORDER_ONLY_VALUE);
        Sync<HardEvent::V_S>();
        for (int g=0;g<16;++g) {
            float a=groupMax.GetValue(g); if (a<1e-4f) a=1e-4f;
            uint32_t bits=GetScalarBitcodeValue<float,uint32_t>(a/448.0f);
            int e=int((bits-1)>>23)-126;
            invScales.SetValue(g,GetScalarBitcodeValue<uint32_t,float>(uint32_t(127-e)<<23));
            words.SetValue(g*8,words.GetValue(g*8)|(uint32_t(e+127)<<((h%4)*8)));
        }
        Sync<HardEvent::S_V>();
        Gather(inv,invScales,indexBuf.Get<uint32_t>()[DIM],0,DIM); VDone();
        Mul(abs,abs,inv,DIM); VDone(); Mins(abs,abs,448.0f,DIM); VDone();
        // E=max(FP32 指数,121)；量化步长为 2^(E-130)，非规格化段固定为 2^-9。
        ShiftRight(temp.ReinterpretCast<uint32_t>(),abs.ReinterpretCast<uint32_t>(),uint32_t(23),DIM); VDone();
        Cast(exp,temp.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DIM); VDone();
        Maxs(exp,exp,121.0f,DIM); VDone();
        Muls(inv,exp,-1.0f,DIM); VDone(); Adds(inv,inv,257.0f,DIM); VDone();
        Cast(temp.ReinterpretCast<int32_t>(),inv,RoundMode::CAST_RINT,DIM); VDone();
        ShiftLeft(temp.ReinterpretCast<uint32_t>(),temp.ReinterpretCast<uint32_t>(),uint32_t(23),DIM); VDone();
        Mul(inv,abs,temp,DIM); VDone();
        Cast(code.ReinterpretCast<int32_t>(),inv,RoundMode::CAST_RINT,DIM); VDone();
        Cast(code,code.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DIM); VDone();
        Adds(exp,exp,-121.0f,DIM); VDone(); Muls(exp,exp,8.0f,DIM); VDone();
        Add(code,code,exp,DIM);
        ShiftRight(temp.ReinterpretCast<uint32_t>(),row.ReinterpretCast<uint32_t>(),uint32_t(31),DIM); VDone();
        Cast(temp,temp.ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,DIM); VDone();
        Muls(temp,temp,128.0f,DIM); VDone(); Add(code,code,temp,DIM); VDone();
        // 码字 0..254 可经 half 精确转成 uint8；符号位也保留负零。
        Cast(rowBuf.Get<half>(),code,RoundMode::CAST_RINT,DIM); VDone();
        auto packed=Work(0).ReinterpretCast<uint8_t>();
        Cast(packed,rowBuf.Get<half>(),RoundMode::CAST_RINT,DIM);
        Sync<HardEvent::V_MTE3>();
        uint64_t base=uint64_t(cfg.query)*HEADS*DIM+(h/8)*4096+(h%8)*32;
        DataCopy(fp8Gm[base],packed,{16,1,0,7});
        if (h%4==3) {
            Sync<HardEvent::S_MTE3>();
            for (int g=0;g<16;++g) {
                int word=(h/8)*32+g*2+(h%8)/4;
                DataCopyPad(scalesGm[uint64_t(word)*scaleStride+cfg.query],words[g*8],{1,4,0,0,0});
            }
        }
        Sync<HardEvent::MTE3_V>(); Sync<HardEvent::MTE3_S>();
    }

    QueryConfig cfg;
    uint32_t taskCount; int32_t scaleStride; bool hasSink,profiling;
    bool nextValid[KV_TILE],valid[2][KV_TILE];
    uint64_t cycles[ProfileCount];
    GlobalTensor<bfloat16_t> qGm;
    GlobalTensor<uint8_t> kv8Gm,kv4Gm,fp8Gm;
    GlobalTensor<int32_t> idsGm,extraIdsGm,configsGm;
    GlobalTensor<float> ropeGm,sinkGm,lseGm,scoresGm,pvGm;
    GlobalTensor<uint32_t> scalesGm;
    GlobalTensor<uint64_t> profileGm;
    TBuf<TPosition::VECCALC> accBuf,workBuf,indexBuf,maskBuf,lutBuf,stateBuf,ropeBuf;
    TBuf<TPosition::VECOUT> arenaBuf,rowBuf;
    TSCM<TPosition::VECOUT,1> qQueue,pQueue;
    TSCM<TPosition::VECOUT,2> kvQueue;
    LocalTensor<bfloat16_t> qL1,pL1,kvL1[2];
};

extern "C" __global__ __aicore__ void mega_attn_v2(
    GM_ADDR q,GM_ADDR kv8,GM_ADDR kv4,GM_ADDR ids,GM_ADDR extraIds,
    GM_ADDR rope,GM_ADDR sink,GM_ADDR configs,uint32_t count,
    GM_ADDR fp8,GM_ADDR scales,GM_ADDR lse,int32_t sfStride,
    GM_ADDR scratch,GM_ADDR profiles,GM_ADDR qkTilingGm,GM_ADDR pvTilingGm,GM_ADDR workspace
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    TPipe pipe; KernelMegaAttentionV2 op; TCubeTiling qk,pv;
    auto a=(__gm__ uint32_t*)qkTilingGm, b=(__gm__ uint32_t*)pvTilingGm;
    for (int i=0;i<sizeof(TCubeTiling)/4;++i) {
        ((uint32_t*)&qk)[i]=a[i]; ((uint32_t*)&pv)[i]=b[i];
    }
    REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),op.qkMm,&qk,op.pvMm,&pv);
    op.Init(q,kv8,kv4,ids,extraIds,rope,sink,configs,count,fp8,scales,lse,sfStride,scratch,profiles,&pipe);
    op.Process();
}

extern "C" void launch_mega_attn_v2(
    const Input &in,const Output &out,const QueryConfig *configs,uint32_t count,uint32_t blocks,
    void *scratch,void *qkTiling,void *pvTiling,void *workspace,void *stream,Profile *profiles
) {
    if (!count) return;
    mega_attn_v2<<<blocks,nullptr,stream>>>(
        (uint8_t*)in.q,(uint8_t*)in.kv_fp8,(uint8_t*)in.kv_fp4,(uint8_t*)in.indices,
        (uint8_t*)in.extra_indices,(uint8_t*)in.cos_sin,(uint8_t*)in.sink,(uint8_t*)configs,count,
        out.fp8,(uint8_t*)out.scales,(uint8_t*)out.lse,out.sf_stride,(uint8_t*)scratch,
        (uint8_t*)profiles,(uint8_t*)qkTiling,(uint8_t*)pvTiling,(uint8_t*)workspace);
}
