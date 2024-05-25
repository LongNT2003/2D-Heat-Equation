// code C for compare result
#include <stdio.h>
#include <malloc.h>
#define N 12
#define c 0.002
#define delta_t 0.05
#define delta_s 0.04
#define Ntime 125

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
    for (int row=0;row<N;row++) {
        for (int col=0;col<N;col++){
            cen=*(T+row*N+col);
            up   = (row==0) ? 25.0 : *(T+ (row-1)*(N) +col);
            down = (row==N-1) ? 25.0 : *(T+ (row+1)*(N) +col);
            right= (col==N-1) ? 100.0 : *(T+ row*(N) +col+1);
            left = (col==0) ? 25.0 : *(T+ row*(N) +col-1);
            *(dT+row*(N)+col) = c*(up+down+left+right-4*cen)/(delta_s*delta_s);
        }    
    }
}
__global__ void SolvingODE(float *T,float *dT) 
{
	int row,col;
        
    for (row=0;row<N;row++) {
        for (col=0;col<N;col++){
            *(T+row*(N)+col) = *(T+row*(N)+col) + delta_t*(*(dT+row*(N)+col));
        }
    }
}
//=========================
int main(int argc, char **argv){
    //1a. Delare and Allocate Mem on CPU
    float *T,*dT;
    T=(float *)malloc((N) * (N) * sizeof(float));
    dT=(float *)malloc((N) * (N) * sizeof(float));
    initData(Tcpu);
    for (int t=0;t<Ntime;t++) {
             Derivative(T,dT);
             SolvingODE(T,dT);
         }
    printData(Tcpu);
    return 0;
}