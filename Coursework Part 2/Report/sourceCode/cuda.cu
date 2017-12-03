__global__ void calcForces(Body *p, int n) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (i < n) {
        /* nested forloop force calculations */
    }
int main() {
    ...
        calcForces<<<BLOCKS,THREADS_PER_BLOCK>>>(d_p, numBodies);
    ...
}