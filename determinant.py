#coding:utf-8
import numpy as np
import time
from videocore.assembler import qpu
from videocore.driver import Driver
import numpy.linalg as LA

@qpu
def kernel(asm):
    
    setup_vpm_write()
    # [メモリ->VPM]
    setup_dma_load(X=0, Y=0, nrows = 16, ncols = 16, mode = '32bit horizontal')
    start_dma_load(uniform)
    wait_dma_load()

    # [VPM->レジスタ] 
    setup_vpm_read(X=0, Y=0, nrows = 16, ncols = 16, mode = '32bit horizontal')
    for i in range(16): # 16×16の行列をすべて読み込む
        mov(ra[i], vpm)
    
    for j in range(16):
        # j行目j列の要素を抽出
        nop() # rotate()の準備
        rotate(r0, ra[j], -j) # j行目j列の要素をレジスタの左へ
        mov(broadcast, r0) # j行目j列の要素をr5にブロードキャスト
        mov(sfu_recip, r5) # j行目j列の要素の逆数を計算->r4
        nop()
        nop()
        fmul(r2, r4, ra[j]) # j行目を、j行目j列の要素で割る
        for i in range(j+1, 16):
            # i行目の左端の要素を抽出
            rotate(r0, ra[i], -j) # i行目j列目の要素をレジスタの左へ
            mov(broadcast, r0) # i行目j列目の要素をr5にブロードキャスト
            fmul(r3, r2, r5) # i行目に、引く数を計算
            fsub(r0, ra[i], r3) # i行目の計算
            mov(ra[i], r0) # 計算した値を元のraレジスタに戻す
    
    for i in range(16):
        mov(vpm,  ra[i])
    
    setup_dma_store(X=0, Y=0, nrows = 16, ncols = 16, mode = '32bit horizontal')
    start_dma_store(uniform)
    wait_dma_store()
    
    exit()

    
with Driver() as drv:
    H=16
    W=16
    A=drv.alloc((H,W),'float32')
    B=drv.alloc((H,W),'float32')
    A[:]=np.round(np.random.rand(H,W)*10, decimals=2)
    B[:]=0.0
    det_CPU = 1
    det_GPU = 1

    start = time.time()
    det_CPU = LA.det(A) #CPUの行列和
    elapsed_cpu = time.time() - start


    start = time.time()
    # Run the program
    drv.execute(
            n_threads=1,
            program=drv.program(kernel),
            uniforms=[A.address, B.address]
    )
    elapsed_gpu = time.time() - start
    for i in range(H):
        det_GPU *= B[i][i]

    elapsed_gpu = time.time() - start
    

    print ("GPU  value:{0}, elapsed_time:{1}".format(det_GPU, elapsed_gpu*1000) + "[msec]")
    print ("CPU  value:{0}, elapsed_time:{1}".format(det_CPU, elapsed_cpu*1000) + "[msec]")
    print(' A '.center(80, '='))
    print(A)

    print(' B '.center(80, '='))
    print(np.round(B, decimals=2))
