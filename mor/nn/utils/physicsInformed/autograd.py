import torch

def jacobian(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]

def hessian(y, x, component=0):
    gradY = torch.autograd.grad(
        y[:, component], x, grad_outputs=torch.ones_like(y[:, component]), 
        create_graph=True, retain_graph=True
    )[0]
    
    hess = []
    for i in range(gradY.shape[1]):
        gradYi = gradY[:, i]
        gradGradYi = torch.autograd.grad(
            gradYi, x, grad_outputs=torch.ones_like(gradYi),
            create_graph=True, retain_graph=True
        )[0]
        hess.append(gradGradYi[:, i:i+1])
    
    return torch.cat(hess, dim=1)

def laplacian(y, x, component=0):
    gradY = torch.autograd.grad(
        y[:, component], x, grad_outputs=torch.ones_like(y[:, component]), 
        create_graph=True, retain_graph=True
    )[0]
    
    lap = 0
    for i in range(gradY.shape[1]):
        gradGradYi = torch.autograd.grad(
            gradY[:, i], x, grad_outputs=torch.ones_like(gradY[:, i]),
            create_graph=True, retain_graph=True
        )[0]
        lap += gradGradYi[:, i:i+1]
    
    return lap
