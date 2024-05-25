// this implement with multiple dimension grid and block
#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#define N 12
#define c 0.002
#define delta_t 0.05
#define delta_s 0.04
#define Ntime 125
#define GridSizeX 2
#define GridSizeY 3
#define BlockSizeX 3
#define BlockSizeY 2
#define ThreadSizeX N/(GridSizeX*BlockSizeX)
#define ThreadSizeY N/(GridSizeY*BlockSizeY)
void initData(float *T){
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            *(T+i*(N)+j)=25.0;
        }
    }

}
void printData(float *T) {
    for (int i = 0; i <= N -1; i++) {
        for (int j = 0; j <= N -1; j++) {
            printf("%6.1f ", *(T + i * (N) + j));
        }
        printf("\n");
    }
}
//=========================
__global__ void Derivative(float *T, float *dT){
    float up,down,left,right,cen;
	int indexX,indexY,row,col,startX,startY,stopX,stopY;
	indexX = blockIdx.x * blockDim.x + threadIdx.x;
	indexY = blockIdx.y * blockDim.y + threadIdx.y;
  	startX = indexX*ThreadSizeX;
	stopX  = startX + ThreadSizeX;
	startY = indexY*ThreadSizeY;
	stopY  = startY + ThreadSizeY;     
    for (row=startX;row<stopX;row++) {
        for (col=startY;col<stopY;col++){
            cen=*(T+row*N+col);
            up   = (row==0) ? 25.0 : *(T+ (row-1)*(N) +col);
            down = (row==N-1) ? 25.0 : *(T+ (row+1)*(N) +col);
            right= (col==N-1) ? 100.0 : *(T+ row*(N) +col+1);
            left = (col==0) ? 25.0 : *(T+ row*(N) +col-1);
            *(dT+row*(N)+col) = c*(up+down+left+right-4*cen)/(delta_s*delta_s);
        }
      __syncthreads();
    }
}
__global__ void SolvingODE(float *T,float *dT) 
{
	int indexX,indexY,row,col,startX,startY,stopX,stopY;
	indexX = blockIdx.x * blockDim.x + threadIdx.x;
	indexY = blockIdx.y * blockDim.y + threadIdx.y;
  	startX = indexX*ThreadSizeX;
	stopX  = startX + ThreadSizeX;
	startY = indexY*ThreadSizeY;
	stopY  = startY + ThreadSizeY;          
    for (row=startX;row<stopX;row++) {
        for (col=startY;col<stopY;col++){
            *(T+row*(N)+col) = *(T+row*(N)+col) + delta_t*(*(dT+row*(N)+col));
        }
    }
	__syncthreads();
}
//=========================
int main(int argc, char **argv){
    //1a. Delare and Allocate Mem on CPU
    float *Tcpu,*dTcpu;
    Tcpu=(float *)malloc((N) * (N) * sizeof(float));
    dTcpu=(float *)malloc((N) * (N) * sizeof(float));
    initData(Tcpu);
    //1b. Delare and Allocate Mem on GPU
    float *Tgpu,*dTgpu;
    cudaMalloc((void**)&Tgpu ,N*N*sizeof(int));
    cudaMalloc((void**)&dTgpu,N*N*sizeof(int));
    //2. Copy Input from CPU to GPU
    cudaMemcpy(Tgpu,Tcpu,N*N*sizeof(int),cudaMemcpyHostToDevice);
    //3. Define Block and Thread Structure
    dim3 dimGrid(GridSizeX,GridSizeY);
    dim3 dimBlock(BlockSizeX,BlockSizeY);
    for (int t=0;t<Ntime;t++) {
             Derivative<<<dimGrid,dimBlock>>>(Tgpu,dTgpu);
             SolvingODE<<<dimGrid,dimBlock>>>(Tgpu,dTgpu);
         }
    //5. Copy Output from GPU to CPU
    cudaMemcpy(Tcpu,Tgpu,N*N*sizeof(int),cudaMemcpyDeviceToHost);
    printData(Tcpu);
    //6. Free Mem on CPU and GPU
    free(Tcpu);free(dTcpu);
    cudaFree(Tgpu);cudaFree(dTgpu);
    return 0;
}