def selectSVDVariant(backendName: str = 'scipy', **kwargs) -> str:
    # Get matrix shape if available
    A = kwargs.get('A')
    if A is not None and hasattr(A, 'shape'):
        n, k = A.shape
        # If n >> k (tall-skinny), dualSVD is often better on GPU
        # because it uses GEMM which is highly optimized
        if n > 5 * k:
            if backendName == 'torch':
                return 'dual'
            else:
                return 'qrSVD' 
        
        # If k > 5 * n (short-fat), we should probably use static SVD 
        # which usually handles the transpose internally
        if k > 5 * n:
            return 'static'
            
        # If both n and k are large, use randomized SVD
        if n > 2000 and k > 2000:
            return 'randomized'
    
    return 'static'
