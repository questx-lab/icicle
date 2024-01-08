#ifndef K_NTT
#define K_NTT
#pragma once

/*

Copyright (c) 2023 Yrrid Software, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the �Software�), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED �AS IS�, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

// #include <stdio.h>
// #include <stdint.h>
// #include "asm.cu"
// #include "Sampled96.cu"

// typedef Sampled96 Math;

#include "thread_ntt.cu"

// EXCHANGE_OFFSET = 129*64*8

#define DATA_OFFSET 0


__device__ uint32_t rev64(uint32_t num, uint32_t nof_digits){
  uint32_t rev_num=0, temp;
  for (int i = 0; i < nof_digits; i++)
  {
    // temp = num & 0x3f;
    // num = num >> 6;
    // rev_num = rev_num << 6;
    temp = num & 0x1f;
    num = num >> 5;
    rev_num = rev_num << 5;
    rev_num = rev_num | temp;
  }
  return rev_num;
  
}

__device__ uint32_t dig_rev(uint32_t num, uint32_t log_size){
  uint32_t rev_num=0, temp, dig_len;
  for (int i = 0; i < 5; i++)
  {
    dig_len = STAGE_SIZES_DEVICE[log_size][i];
    temp = num & ((1<<dig_len)-1);
    num = num >> dig_len;
    rev_num = rev_num << dig_len;
    rev_num = rev_num | temp;
  }
  return rev_num;
  
}

__launch_bounds__(64)
__global__ void reorder64_kernel(uint4* arr, uint4* arr_reordered, uint32_t log_size){
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t rd = tid;
  // uint32_t wr = stride * threadIdx.x + blockIdx.x;
  // uint32_t nof_digits = 4; //todo - parametrize
  // uint32_t wr = rev64(tid, nof_digits);
  uint32_t wr = dig_rev(tid, log_size);
  arr_reordered[wr] = arr[rd];
  // arr_reordered[wr + (1<<(nof_digits*6))] = arr[rd + (1<<(nof_digits*6))];
  arr_reordered[wr + (1<<log_size)] = arr[rd + (1<<log_size)];
  // arr_reordered[wr + (1<<(nof_digits*5))] = arr[rd + (1<<(nof_digits*5))];
  // arr_reordered[wr + 64 * stride] = arr[rd + 64 * stride];
  // if (tid == 1647) printf("%d %d\n", rd,wr);
}

// void reorder64(uint4* arr, uint32_t n, uint32_t logn)
// {
//   uint4* arr_reorderd;
//   cudaMalloc(&arr_reorderd, n * 2 * sizeof(uint4));
//   int number_of_threads = MAX_THREADS_BATCH;
//   int number_of_blocks = (n * 2 + number_of_threads - 1) / number_of_threads;
//   reorder64_kernel<<<number_of_blocks, number_of_threads>>>(arr, arr_reorderd, n);
//   cudaMemcpyAsync(arr, arr_reorderd, n * batch_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
//   cudaFreeAsync(arr_reversed, stream);
// }

__launch_bounds__(384)
__global__ void ntt1024(uint64_t* out, uint64_t* in, uint32_t* next, uint32_t count) {
  NTTEngine32 engine;
  uint32_t    dataIndex=0;

  #ifdef COMPUTE_ONLY
    bool        first=true;
  #endif
  
  engine.initializeRoot();
    
  while(true) {
    if((threadIdx.x & 0x1F)==0)
      dataIndex=atomicAdd(next, 1);
    dataIndex=__shfl_sync(0xFFFFFFFF, dataIndex, 0);  //send value to all threads in warp    
    if(dataIndex<count) {
      #if defined(COMPUTE_ONLY)
        if(first)
          engine.loadGlobalData(in, dataIndex);
        first=false;
      #else
        engine.loadGlobalData(in, dataIndex);
      #endif
    }
    else {
      if(dataIndex==count + (gridDim.x*blockDim.x>>5) - 1) { //didn't understand condition
        // last one to finish, reset the counter
        atomicExch(next, 0);
      }
      return;
    }
    #pragma unroll 1
    for(uint32_t phase=0;phase<2;phase++) {
      // ntt32 produces a lot of instructions, so we put this in a loop
      engine.ntt32(); 
      if(phase==0) {
        engine.storeSharedData(DATA_OFFSET);
        __syncwarp(0xFFFFFFFF);
        engine.loadSharedDataAndTwiddle32x32(DATA_OFFSET);
      }
    }
    engine.storeGlobalData(out, dataIndex);
  }
}


__global__ void thread_ntt_kernel(test_scalar* out, test_scalar* in, uint32_t* next, uint32_t count) {
  NTTEngine engine;
  uint32_t    dataIndex=blockIdx.x*blockDim.x+threadIdx.x;

  #ifdef COMPUTE_ONLY
    bool        first=true;
  #endif
  
  // engine.initializeRoot();
  engine.loadGlobalDataDep(in, dataIndex);
  // engine.ntt4_4();
  // for (int i = 0; i < 16; i++)
  // {
  //   engine.X[i] = in[i];
  // }
  

  // for (int i = 0; i < 100; i++)
  // {
    // engine.ntt16win_lowreg();
    // engine.ntt8_2();
    // engine.ntt8_2();
    // engine.ntt16();
    // engine.ntt16_win8ct2();
    engine.ntt16win();
    // engine.X[2] = engine.X[2]*engine.X[0];
  // }
  // engine.ntt16();
  // engine.X[0] = engine.X[0] + engine.X[1];
  // out[0] = out[0] + out[1];
  // out[0] = test_scalar::zero();
  // engine.storeGlobalData(out, dataIndex);
  // engine.storeGlobalData8_2(out, dataIndex);
  // engine.storeGlobalData16(out, dataIndex);
    
  // while(true) {
  //   if((threadIdx.x & 0x1F)==0)
  //     dataIndex=atomicAdd(next, 1);
  //   dataIndex=__shfl_sync(0xFFFFFFFF, dataIndex, 0);      
  //   if(dataIndex<count) {
  //     #if defined(COMPUTE_ONLY)
  //       if(first)
  //         engine.loadGlobalData(in, dataIndex);
  //       first=false;
  //     #else
  //       engine.loadGlobalData(in, dataIndex);
  //     #endif
  //   }
  //   else {
  //     if(dataIndex==count + (gridDim.x*blockDim.x>>5) - 1) {
  //       // last one to finish, reset the counter
  //       atomicExch(next, 0);
  //     }
  //     return;
  //   }
  //   #pragma unroll 1
  //   for(uint32_t phase=0;phase<2;phase++) {
  //     // ntt32 produces a lot of instructions, so we put this in a loop
  //     engine.ntt32(); 
  //     if(phase==0) {
  //       engine.storeSharedData(DATA_OFFSET);
  //       __syncwarp(0xFFFFFFFF);
  //       engine.loadSharedDataAndTwiddle32x32(DATA_OFFSET);
  //     }
  //   }
    engine.storeGlobalDataDep(out, dataIndex);
  // }
}

__launch_bounds__(64)
// __global__ void ntt_kernel_split_transpose(test_scalar* out, test_scalar* in) {
__global__ void ntt_kernel_split_transpose(uint4* out, uint4* in) {
  NTTEngine engine;
  uint32_t    dataIndex=blockIdx.x*blockDim.x+threadIdx.x;

  // if (blockIdx.x !=1) return;

  // if (blockIdx.x ==0 && threadIdx.x ==0) printf("start kernel\n");  
  // __shared__ uint4 shmem[2048*3];
  extern __shared__ uint4 shmem[];
  // if (blockIdx.x ==0 && threadIdx.x ==0) printf("shmem\n");

  // #ifdef COMPUTE_ONLY
  //   bool        first=true;
  // #endif
  
  // engine.initializeRoot();
  // engine.loadGlobalData(in, dataIndex);
  // engine.ntt4_4();
  // for (int i = 0; i < 100000; i++)
  // {
    // engine.ntt16win();
    // engine.ntt16win_lowreg();
    // engine.ntt8_2();
    // engine.ntt8_2();
    // engine.ntt16();
  // engine.ntt16_win8ct2();
  // }
  // engine.ntt16();
  // engine.X[0] = engine.X[0] + engine.X[1];
  // out[0] = out[0] + out[1];
  // out[0] = test_scalar::zero();
  // engine.storeGlobalData(out, dataIndex);
  // engine.storeGlobalData8_2(out, dataIndex);
  // engine.storeGlobalData16(out, dataIndex);
    
  // while(true) {
    // if((threadIdx.x & 0x1F)==0)
    //   dataIndex=atomicAdd(next, 1);
    // dataIndex=__shfl_sync(0xFFFFFFFF, dataIndex, 0);      
    // if(dataIndex<count) {
    //   #if defined(COMPUTE_ONLY)
    //     if(first)
    //       engine.loadGlobalData(in, dataIndex);
    //     first=false;
    //   #else
    //     engine.loadGlobalData(in, dataIndex);
    //   #endif
    // }
    // else {
    //   if(dataIndex==count + (gridDim.x*blockDim.x>>5) - 1) {
    //     // last one to finish, reset the counter
    //     atomicExch(next, 0);
    //   }
    //   return;
    // }
    // if (threadIdx.x!=0) return;
    // engine.loadGlobalDataDep(in, dataIndex);
    // engine.loadGlobalData(in,blockIdx.x*512*2,1,64*8); //todo - change function to fit global ntt
    // engine.loadGlobalData(in,blockIdx.x*512*2,1,256*8); //todo - change function to fit global ntt
    // __syncthreads();
    // if (blockIdx.x ==0 && threadIdx.x ==0) printf("load global\n");
    // engine.externalTwiddles(); //todo
    // engine.twiddles256();
    // engine.ntt16_win8ct2();
    // engine.twiddles256();
    // engine.ntt16_win8ct2();
    // #pragma unroll 1
    // for (uint32_t i=0;i<100;i++) {
    #pragma unroll 1
    for (uint32_t phase=0;phase<2;phase++) {
      // ntt32 produces a lot of instructions, so we put this in a loop
      // engine.ntt16_win8ct2();
      // engine.plus();
      // engine.twiddles256();
      // engine.load_twiddles(threadIdx.x&0x7);
      // engine.twiddles64(threadIdx.x&0x7);
      // engine.ntt16_win8ct2();
      // engine.ntt8win();
      // engine.ntt16_win8ct2();
      // engine.twiddles256();
      // engine.ntt16_win8ct2();
      // engine.ntt16win();
      // engine.twiddles256();
      // engine.ntt16();
      // if(phase==0) {
      //   engine.SharedDataColumns2(shmem, true, false); //store low
      //   __syncthreads();
      //   // if (blockIdx.x ==0 && threadIdx.x ==0) printf("store shmem low\n");
      //   // if (blockIdx.x ==0 && threadIdx.x ==0){
      //   //   for (int i = 0; i < 512; i++)
      //   //   {
      //   //     if (i%32==0) printf("\n");
      //   //     if (i%256==0) printf("\n");
      //   //     printf("%d, ",shmem[i].w);
      //   //   }
      //   // }
      //   // __syncthreads();
      //   engine.SharedDataRows2(shmem, false, false); //load low
      //   // if (blockIdx.x ==0 && threadIdx.x ==0) printf("load shmem low\n");
      //   // __syncthreads(); //can avoid with switching rows and columns
      //   // if (blockIdx.x ==0 && threadIdx.x ==1){
      //   //   for (int i = 0; i < 16; i++)
      //   //   {
      //   //     printf("\n");
      //   //     printf("%d, ",engine.X[i].limbs_storage.limbs[0]);
      //   //   }
      //   // }
      //   engine.SharedDataRows2(shmem, true, true); //store high
      //   __syncthreads();
      //   // if (blockIdx.x ==0 && threadIdx.x ==0) printf("store shmem high\n");
      //   // if (blockIdx.x ==0 && threadIdx.x ==0){
      //   //   for (int i = 0; i < 2048; i++)
      //   //   {
      //   //     if (i%16==0) printf("\n");
      //   //     if (i%256==0) printf("\n");
      //   //     printf("%d, ",shmem[i].w);
      //   //   }
      //   // }
      //   // __syncthreads();
      //   engine.SharedDataColumns2(shmem, false, true); //load high
      //   // if (blockIdx.x ==0 && threadIdx.x ==0) printf("load shmem high\n");
      //   // engine.twiddles256();
      //   // engine.ntt16_win8ct2();
      //   // engine.twiddles256();
      // }
    // }
    }
    // #pragma unroll 1
    // for (uint32_t i=0;i<100*2;i++) {
    // // #pragma unroll 1
    // // for (uint32_t phase=0;phase<2;phase++) {
    //   // ntt32 produces a lot of instructions, so we put this in a loop
    //   // engine.ntt16_win8ct2();
    //   // if (i%2) engine.twiddles256();
    //   engine.twiddles256();
    //   engine.ntt16_win8ct2();
    //   // engine.ntt16win();
    // }
    // engine.storeGlobalData(out,blockIdx.x*512*2,1,64*8); //todo - change function to fit global ntt
    // engine.storeGlobalData(out,blockIdx.x*512*2,1,256*8); //todo - change function to fit global ntt
    // engine.storeGlobalDataDep(out, dataIndex); //todo - change function to fit global ntt
  // }
}


__launch_bounds__(64)
__global__ void ntt64(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {
// __global__ void ntt64(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint32_t log_size, uint32_t tw_log_size, uint32_t data_stride, uint32_t twiddle_stride) {
// __global__ void ntt64(uint4* in, uint4* out, uint4* twiddles, uint32_t log_size, uint32_t tw_log_size, uint32_t data_stride, uint32_t twiddle_stride) {
  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 8;
  s_meta.ntt_block_size = 64;
  s_meta.ntt_block_id = (blockIdx.x << 3) + (strided? (threadIdx.x & 0x7) : (threadIdx.x >> 3));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 3) : (threadIdx.x & 0x7);
  // s_meta.ntt_meta_block_size = 1 << (6*(stage_num+1)); //last stage is 0
  // s_meta.ntt_meta_block_id = (s_meta.ntt_block_id >> (6*stage_num)) % s_meta.ntt_meta_block_size;
  // s_meta.ntt_meta_inp_id = s_meta.ntt_inp_id + blockDim.x*(s_meta.ntt_block_id % (1 << (6*stage_num)));

  // if (twiddle_stride && threadIdx.x>15) return;
  
  // engine.initializeRoot(strided);
  engine.loadBasicTwiddles(inv);
  engine.loadInternalTwiddles(internal_twiddles, strided);
    
  // #pragma unroll 1
  // for (int i = 0; i < 260; i++) //todo - function of size
  // {
  engine.loadGlobalData(in, data_stride, log_size, strided, s_meta);
  if (twiddle_stride && dit) {
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
    engine.loadExternalTwiddles(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.WE[i]);
    //   }
    //   printf("\n");
    // }
    engine.twiddlesExternal();
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
  }
    // engine.storeGlobalData(in, data_stride, log_size, strided, s_meta);
    // return;


    // if (twiddle_stride){
    //   engine.loadGlobalData(in,blockIdx.x*64*8,data_stride,1<<log_size); //todo - parametize
    //   engine.loadExternalTwiddles(twiddles,blockIdx.x*64*8,twiddle_stride,1<<tw_log_size); //todo - parametize
    //   engine.twiddlesExternal();
    // }
    // else{
    //   engine.loadGlobalData(in,blockIdx.x*8,data_stride,1<<log_size); //todo - parametize
    // }
    // engine.twiddles64();

    #pragma unroll 1
    for(uint32_t phase=0;phase<2;phase++) {
      // this code produces a lot of instructions, so we put this in a loop
      engine.ntt8win(); 
      if(phase==0) {
        engine.SharedDataColumns2(shmem, true, false, strided); //store low
        __syncthreads();
        engine.SharedDataRows2(shmem, false, false, strided); //load low
        engine.SharedDataRows2(shmem, true, true, strided); //store high
        __syncthreads();
        engine.SharedDataColumns2(shmem, false, true, strided); //load high
        engine.twiddles64();
      }
    }
    // if (twiddle_stride) engine.twiddlesExternal();
    if (twiddle_stride && !dit) {
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
    engine.loadExternalTwiddles(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num);
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.WE[i]);
    //   }
    //   printf("\n");
    // }
    // if (twiddle_stride>64) engine.twiddlesExternal();
    engine.twiddlesExternal();
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
  }

    // engine.storeGlobalData(twiddle_stride? out :in, data_stride, log_size);
    engine.storeGlobalData(in, data_stride, log_size, strided, s_meta);
  // }
}

__launch_bounds__(64)
__global__ void ntt32(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint32_t log_size, uint32_t data_stride, uint32_t twiddle_stride, bool strided, uint32_t stage_num, bool inv, bool dit) {
// __global__ void ntt64(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint32_t log_size, uint32_t tw_log_size, uint32_t data_stride, uint32_t twiddle_stride) {
// __global__ void ntt64(uint4* in, uint4* out, uint4* twiddles, uint32_t log_size, uint32_t tw_log_size, uint32_t data_stride, uint32_t twiddle_stride) {
  NTTEngine engine;
  stage_metadata s_meta;
  extern __shared__ uint4 shmem[];

  s_meta.th_stride = 4;
  s_meta.ntt_block_size = 32;
  s_meta.ntt_block_id = (blockIdx.x << 4) + (strided? (threadIdx.x & 0xf) : (threadIdx.x >> 2));
  s_meta.ntt_inp_id = strided? (threadIdx.x >> 4) : (threadIdx.x & 0x3);
  // s_meta.ntt_meta_block_size = 1 << (6*(stage_num+1)); 
  // s_meta.ntt_meta_block_id = (s_meta.ntt_block_id >> (6*stage_num)) % s_meta.ntt_meta_block_size; 
  // s_meta.ntt_meta_inp_id = s_meta.ntt_inp_id + blockDim.x*(s_meta.ntt_block_id % (1 << (6*stage_num)));

  // if (twiddle_stride && threadIdx.x>15) return;
  
  // engine.initializeRoot(strided);
  engine.loadBasicTwiddles(inv);
  engine.loadInternalTwiddles32(internal_twiddles, strided);
    
  // #pragma unroll 1
  // for (int i = 0; i < 260; i++) //todo - function of size
  // {
  engine.loadGlobalData(in, data_stride, log_size, strided, s_meta);
  if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i = 0; i < 8; i++)
      {
        printf("%d, ",engine.X[i]);
      }
      printf("\n");
    }
  if (twiddle_stride && dit) {
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
    engine.loadExternalTwiddles32(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num); //TODO
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.WE[i]);
    //   }
    //   printf("\n");
    // }
    engine.twiddlesExternal(); //TODO
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
  }
    // engine.storeGlobalData(in, data_stride, log_size, strided, s_meta);
    // return;


    // if (twiddle_stride){
    //   engine.loadGlobalData(in,blockIdx.x*64*8,data_stride,1<<log_size); //todo - parametize
    //   engine.loadExternalTwiddles(twiddles,blockIdx.x*64*8,twiddle_stride,1<<tw_log_size); //todo - parametize
    //   engine.twiddlesExternal();
    // }
    // else{
    //   engine.loadGlobalData(in,blockIdx.x*8,data_stride,1<<log_size); //todo - parametize
    // }
    // engine.twiddles64();

    // #pragma unroll 1
    // for(uint32_t phase=0;phase<2;phase++) {
      // this code produces a lot of instructions, so we put this in a loop
      engine.ntt8win(); 
      if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i = 0; i < 8; i++)
      {
        printf("%d, ",engine.X[i]);
      }
      printf("\n");
    }
      // if(phase==0) {
      engine.twiddles32();
      if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i = 0; i < 8; i++)
      {
        printf("%d, ",engine.X[i]);
      }
      printf("\n");
    }
      engine.SharedData32Columns8(shmem, true, false, strided); //store low
      __syncthreads();
      // if (blockIdx.x ==0 && threadIdx.x ==0){
      //     for (int i = 0; i < 64; i++)
      //     {
      //       if (i%4==0) printf("\n");
      //       if (i%32==0) printf("\n");
      //       printf("%d, ",shmem[i].w);
      //     }
      //     printf("\n");
      //   }
      // __syncthreads();
      engine.SharedData32Rows4_2(shmem, false, false, strided); //load low
      engine.SharedData32Rows8(shmem, true, true, strided); //store high
      __syncthreads();
      // if (blockIdx.x ==0 && threadIdx.x ==0){
      //     for (int i = 0; i < 64; i++)
      //     {
      //       if (i%8==0) printf("\n");
      //       if (i%32==0) printf("\n");
      //       printf("%d, ",shmem[i].w);
      //     }
      //     printf("\n");
      //     printf("\n");
      //   }
      //   __syncthreads();
      engine.SharedData32Columns4_2(shmem, false, true, strided); //load high
      // if (blockIdx.x == 0 && threadIdx.x == 4){
      // for (int i = 0; i < 8; i++)
      // {
      //   printf("%d, ",engine.X[i]);
      // }
      // printf("\n");
      // }
      engine.ntt4_2();
      if (blockIdx.x == 0 && threadIdx.x == 0){
      for (int i = 0; i < 8; i++)
      {
        printf("%d, ",engine.X[i]);
      }
      printf("\n");
    }
      // }
    // }
    // if (twiddle_stride) engine.twiddlesExternal();
    if (twiddle_stride && !dit) {
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
    engine.loadExternalTwiddles32(twiddles, twiddle_stride, strided, s_meta, log_size, stage_num); //TODO
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.WE[i]);
    //   }
    //   printf("\n");
    // }
    // if (twiddle_stride>64) engine.twiddlesExternal();
    engine.twiddlesExternal(); //TODO
    // if (blockIdx.x == 8 && threadIdx.x == 8){
    //   for (int i = 0; i < 8; i++)
    //   {
    //     printf("%d, ",engine.X[i]);
    //   }
    //   printf("\n");
    // }
  }

    // engine.storeGlobalData(twiddle_stride? out :in, data_stride, log_size);
    engine.storeGlobalData32(in, data_stride, log_size, strided, s_meta);
  // }
}

// __global__ void generate_base_tables(test_scalar* w6_table, test_scalar* w12_table, test_scalar* w18_table){

// test_scalar w, t;

// switch (threadIdx.x)
// {
// case 0:
//   w = test_scalar::omega(6);
//   t = test_scalar::one();
//   for (int i = 0; i < 64; i++)
//   {
//     // w6_table[i] = t.load_half(false);
//     // w6_table[i + 64] = t.load_half(true);
//     w6_table[i] = t;
//     t = t * w;
//   }
//   break;
// case 1:
//   w = test_scalar::omega(12);
//   t = test_scalar::one();
//   for (int i = 0; i < 64; i++)
//   {
//     w12_table[i] = t;
//     t = t * w;
//   }
//   break;
// case 2:
//   w = test_scalar::omega(18);
//   t = test_scalar::one();
//   for (int i = 0; i < 64; i++)
//   {
//     w18_table[i] = t;
//     t = t * w;
//   }
//   break;

// default:
//   break;
// }

// }

__global__ void generate_base_table(int x, uint4* base_table, bool inv, test_scalar norm_factor){
  
  test_scalar w = inv? test_scalar::omega_inv(x) : test_scalar::omega(x);
  test_scalar t = test_scalar::one(), temp;
  for (int i = 0; i < 64; i++)
  {
    // base_table[i] = t;
    temp = t * norm_factor;
    base_table[i] = temp.load_half(false);
    base_table[i+64] = temp.load_half(true);
    t = t * w;
  }
}

__global__ void generate_twiddle_combinations(uint4* w6_table, uint4* w12_table, uint4* w18_table, uint4* w24_table, uint4* w30_table, uint4* twiddles, uint32_t log_size, uint32_t stage_num, test_scalar norm_factor){
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // int pow = stage_num*6;
  // int exp = ((tid & ((1<<pow)-1)) * (tid >> pow)) << (24-pow);
  int range1 = 0, range2 = 0, ind;
  for (ind = 0; ind < stage_num; ind++) range1 += STAGE_SIZES_DEVICE[log_size][ind];
  range2 = STAGE_SIZES_DEVICE[log_size][ind];
  int root_order = range1 + range2;
  int exp = ((tid & ((1<<range1)-1)) * (tid >> range1)) << (30-root_order);
  // int exp = ((tid & 0x3f) * (tid >> 6)) << (18-pow);
  test_scalar w6,w12,w18,w24,w30;
  w6.store_half(w6_table[exp >> 24], false);
  w6.store_half(w6_table[(exp >> 24) + 64], true);
  w12.store_half(w12_table[((exp >> 18) & 0x3f)], false);
  w12.store_half(w12_table[((exp >> 18) & 0x3f) + 64], true);
  w18.store_half(w18_table[((exp >> 12) & 0x3f)], false);
  w18.store_half(w18_table[((exp >> 12) & 0x3f) + 64], true);
  w24.store_half(w24_table[((exp >> 6) & 0x3f)], false);
  w24.store_half(w24_table[((exp >> 6) & 0x3f) + 64], true);
  w30.store_half(w30_table[(exp & 0x3f)], false);
  w30.store_half(w30_table[(exp & 0x3f) + 64], true);
  // test_scalar t = w6_table[exp >> 18] * w12_table[(exp >> 12) & 0x3f] * w18_table[(exp >> 6) & 0x3f] * w24_table[exp & 0x3f];
  // test_scalar t = w6_table[exp >> 12] * w12_table[(exp >> 6) & 0x3f] * w18_table[exp & 0x3f];
  // test_scalar t = w6_table[exp >> 6] * w12_table[exp & 0x3f];
  test_scalar t = w6 * w12 * w18 * w24 * w30 * norm_factor;
  // twiddles[tid + tw_offsets[2*stage_num]] = t.load_half(false);
  // twiddles[tid + tw_offsets[2*stage_num+1]] = t.load_half(true);
  twiddles[tid + LOW_W_OFFSETS[log_size][stage_num]] = t.load_half(false);
  twiddles[tid + HIGH_W_OFFSETS[log_size][stage_num]] = t.load_half(true);
  // twiddles[tid] = t.load_half(false);
  // twiddles[tid + (1<<(pow+6))] = t.load_half(true);
  // twiddles[tid + (1<<18)] = t.load_half(true);
  // twiddles[tid + (1<<12)] = t.load_half(true);

}
// __global__ void generate_internal_twiddles(uint4* twiddles){
//   test_scalar w = test_scalar::omega(6);
//   test_scalar t = test_scalar::one();
//   for (int i = 0; i < 64; i++)
//   {
//     twiddles[i] = t.load_half(false);
//     twiddles[i+64] = t.load_half(true);
//     t = t * w;
//   }
// }

uint4* generate_external_twiddles(uint4* twiddles, uint32_t log_size, bool inv){

  uint4* w6_table;
  uint4* w12_table;
  uint4* w18_table;
  uint4* w24_table;
  uint4* w30_table;
  cudaMalloc((void**)&w6_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w12_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w18_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w24_table, sizeof(uint4)*64*2);
  cudaMalloc((void**)&w30_table, sizeof(uint4)*64*2);
  generate_base_table<<<1,1>>>(6, w6_table, inv, (log_size == 6 && inv)? test_scalar::inv_log_size(6) : test_scalar::one()); //TODO - generalize inv
  generate_base_table<<<1,1>>>(12, w12_table, inv, test_scalar::one());
  generate_base_table<<<1,1>>>(18, w18_table, inv, test_scalar::one());
  generate_base_table<<<1,1>>>(24, w24_table, inv, test_scalar::one());
  generate_base_table<<<1,1>>>(30, w30_table, inv, test_scalar::one());
  // for (int i = 1; i < log_size/6; i++)
  // {
  //   generate_twiddle_combinations<<<16<<((i-1)*6),256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles, log_size, i, ((i+1)*6 == log_size && inv)? test_scalar::inv_log_size(log_size) : test_scalar::one());
  // }
  uint32_t temp = STAGE_SIZES_HOST[log_size][0];
  for (int i = 1; i < 5; i++)
  {
    // printf("log size %d\n",log_size);
    // printf("ergwrtheryhetyjetyjthert %d\n",temp);
    if (!STAGE_SIZES_HOST[log_size][i]) break;
    temp += STAGE_SIZES_HOST[log_size][i];
    generate_twiddle_combinations<<<1<<(temp-8),256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles, log_size, i, ((i+1)*6 == log_size && inv)? test_scalar::inv_log_size(log_size) : test_scalar::one()); //TODO - generalize inv
  }
  

  // generate_twiddle_combinations<<<16,256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles,1);
  // generate_twiddle_combinations<<<16*64,256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles,2);
  // generate_twiddle_combinations<<<1024*64,256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles,3);
  // generate_twiddle_combinations<<<1024*64*64,256>>>(w6_table, w12_table, w18_table, w24_table, w30_table, twiddles,4);
  return w6_table;
}

void new_ntt(uint4* in, uint4* out, uint4* twiddles, uint4* internal_twiddles, uint32_t log_size, bool inv, bool dit){
// void new_ntt(uint4* in, uint4* out, uint4* twiddles, uint32_t log_size){
  if (log_size==5){
    ntt32<<<1,4,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
    return;
  }
  // if (log_size==10){
  //   ntt32<<<2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 32, 32, true, 1, inv, dit);
  //   ntt32<<<2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  // if (log_size==11){
  //   ntt32<<<2*2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 64, true, 1, inv, dit);
  //   ntt64<<<2*2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  // if (log_size==16){
  //   ntt32<<<2*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*32, 64*32, true, 2, inv, dit); //stride is the size of the next meta block
  //   ntt32<<<2*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 64, true, 1, inv, dit);
  //   ntt64<<<2*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  // if (log_size==17){
  //   ntt32<<<2*64*2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*64, 64*64, true, 2, inv, dit);
  //   ntt64<<<2*64*2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 64, true, 1, inv, dit);
  //   ntt64<<<2*64*2,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  // if (log_size==15){
  //   ntt32<<<2*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 32*32, 32*32, true, 2, inv, dit);
  //   ntt32<<<2*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 32, 32, true, 1, inv, dit);
  //   ntt32<<<2*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  // if (log_size==20){
  //   ntt32<<<2*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 32*32*32, 0, true, 3, inv, dit);
  //   // ntt32<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*32, 0, true, 2, inv, dit);
  //   // ntt32<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 0, true, 1, inv, dit);
  //   // ntt64<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  // if (log_size==21){
  //   ntt32<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*32*32, 0, true, 3, inv, dit);
  //   // ntt32<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*32, 0, true, 2, inv, dit);
  //   // ntt32<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 0, true, 1, inv, dit);
  //   // ntt64<<<4*32*32,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
  //   return;
  // }
  for (int i = 4; i >= 0; i--)
  {
    uint32_t stage_size = STAGE_SIZES_HOST[log_size][i];
    uint32_t stride_log = 0;
    for (int j=0; j<i; j++) stride_log += STAGE_SIZES_HOST[log_size][j];
    printf(" %d %d %d sdfgsdfg\n", i,stage_size, stride_log);
    if (stage_size == 6) ntt64<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1<<stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
    if (stage_size == 5) ntt32<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1<<stride_log, i? (1<<stride_log) : 0, i, i, inv, dit);
  }
  return;

  uint32_t nof_stages = log_size/6;
  if (nof_stages==1){
    ntt64<<<1,8,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0, inv, dit);
    return;
  }
  // if (nof_stages==2){
  //   ntt64<<<8,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 0, false, 0);
  //   ntt64<<<8,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0);
  //   return;
  // }
  if (dit){
    for (int i = 0; i < nof_stages; i++)
    {
      ntt64<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1<<(i*6), i? 1<<(i*6) : 0, i, i, inv, dit);
    }
    return;
  }
  for (int i = nof_stages-1; i >= 0; i--)
  {
    ntt64<<<1<<(log_size-9),64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1<<(i*6), i? 1<<(i*6) : 0, i, i, inv, dit);
  }
  // if (log_size == 6){
  // ntt64<<<1,8,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0);
  // }
  // if (log_size == 12){
  // ntt64<<<8,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 0, true, 1);
  // ntt64<<<8,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 1, false, 0);
  // }
  // if (log_size == 18){
  // ntt64<<<8*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*64, 64*64, true, 2);
  // // cudaDeviceSynchronize();
  // // printf("cuda err %d\n",cudaGetLastError());
  // ntt64<<<8*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 64, true, 1);
  // // cudaDeviceSynchronize();
  // // printf("cuda err %d\n",cudaGetLastError());
  // ntt64<<<8*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0);
  // // cudaDeviceSynchronize();
  // // printf("cuda err %d\n",cudaGetLastError());
  // }
  // if (log_size == 24){
  // ntt64<<<8*64*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*64*64, 64*64*64, true, 3);
  // ntt64<<<8*64*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64*64, 64*64, true, 2);
  // ntt64<<<8*64*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 64, 64, true, 1);
  // ntt64<<<8*64*64,64,8*64*sizeof(uint4)>>>(in, out, twiddles, internal_twiddles, log_size, 1, 0, false, 0);
  // }
}

#endif