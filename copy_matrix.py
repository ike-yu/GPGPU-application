#coding:utf-8
import numpy as np
import time
from videocore.assembler import qpu
from videocore.driver import Driver

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
    A[:]=np.random.randn(H,W)
    B[:]=0.0

    start = time.time()
    CC=A+B #CPUの行列和
    elapsed_cpu = time.time() - start


    start = time.time()
    # Run the program
    drv.execute(
            n_threads=1,
            program=drv.program(kernel),
            uniforms=[A.address, B.address]
    )
    elapsed_gpu = time.time() - start
    

    print ("GPU:elapsed_time:{0}".format(elapsed_gpu*1000) + "[msec]")
    print ("CPU:elapsed_time:{0}".format(elapsed_cpu*1000) + "[msec]")
    print(' A '.center(80, '='))
    print(A)

    print(' B '.center(80, '='))
    print(B)

    # print('maximum absolute error: {:.4e}'.format(float(np.max(np.abs(C - CC)))))
