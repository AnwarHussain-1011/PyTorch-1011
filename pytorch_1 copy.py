import torch

# Initializing Tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor. dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#other common onitializations methods
x = torch.empty((3, 3))
x = torch.zeros((3, 3))
x = torch.rand((3,3))
x = torch.eye(5,5) #i, eye
x = torch.arange(start=0, end=5, step=1)
x = torch.linspace(start=0.1, end=1, steps=10)
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) #normal distribution
x = torch.diag(torch.ones(3)) #diagonal matrix)
x = torch.rand(1, 0) #uniform distribution between 0 and 1

# how to initialize from existing data
tensor = torch.arange(4)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

# array to tensor conversion and vice versa
import numpy as np # Must be at the top of the file!

# Line 37 (The fix): Must use the NumPy alias 'np' to create the array
np_array = np.zeros((5,5)) 

# Line 38 (The line that was failing): Now it will work
tensor = torch.from_numpy(np_array) 

# Line 39 (Next step): Convert the PyTorch Tensor back to a NumPy array
nparray_back = tensor.numpy()

# Line 40 (Output): Prints the types, confirming success
print(type(np_array))
print(type(nparray_back))


#=====================================================================
        #Tensor Math and comparison operations
#=====================================================================
x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])
# Basic operations
#Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)

z2 = torch.add(x,y)
z = x + y
#subtraction
z = x - y
#multiplication
z = x * y
#division
z = torch.true_divide(x, y)
# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x # t =t+x
#exponentiation
z = x ** 2

# simple comparision
z = x >0
z = x <0

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2) # 2x3
x3 = x1.mm(x2)

#matrix exponentiation
matrix_exp = torch.rand(5, 5)
print(matrix_exp.matrix_power(3))

# Element wise exponentiation
z = x *y
print(z)

# dot product
z = torch.dot(x,y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # Batch, n,p)

#example of broadcasting
x1 = torch.rand ((5,5))
z2 = torch.rand((1,5))

z = x1 - z2
z = x1 ** z2

#other useful tensor operations
sum_x = torch.sum(x, dim=0)
values, indices = torch.max(x, dim=0) #dim=0 means column wise, dim=1 means row wise
values, indices = torch.min(x, dim=0)
abs_x = torch.mean(x.float())
z = torch.argmin(x, dim=0)
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y, dim=0, descending=False)

z = torch.clamp(x, min=0)

x = torch.tensor([1,0,1,1,1], dtype=torch.bool)
z = torch.any(x)
#print(z)
# or 
z = torch.all(x) #false
print(z)



# tensor indexing and slicing
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0].shape) # x{0,:]

print(x[:, 0].shape)

print (x[2, 0:10]) #0:10- [0,1,2,3,4,5,6,7,8,9]

x[0, 0] = 100

#fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.tensor((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
x = torch.arange(10).reshape(2, 5) # Creates a 2x5 matrix
print(x[rows, cols].shape) #2 elements tensor

#more advanced indexing
x = torch.arange(10)
print(x[(x < 2) & (x > 8)])
print(x[x.remainder(2) == 0])

# useful tensor operations
print(torch.where(x > 5, x, x*2))
print(x.ndimension()) #5x5x5
print(x.numel()) #number of elements

 #Tensor Reshaping and Manipulation


x = torch.arange(9)

x_3x3 = x.view(3,3)
print(x_3x3)
x_3x3 + x.reshape(3, 3)

y = x_3x3.t()
#Pint(y)
print(y.contiguous().view(9))

x1 = torch.rand(2, 5)
x2 = torch.rand(2, 5)
print(torch.cat((x1, x2), dim=0).shape) # along rows
print(torch.cat((x1, x2), dim=1).shape) # along columns

z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1)
print(z.shape)

z = x.permute(0, 2, 1) # similar to transpose but more general
print(z.shape)

x = torch.arange(10) #[10]
print(x.unsqueeze(0).shape) #[1,10]
print (x.unsqueeze(1).shape) #[10,1]

# Create a 1D tensor, then use two unsqueeze calls to make it 3D (e.g., [1, 10, 1])
x = torch.arange(10).unsqueeze(0).unsqueeze(2) 
# Now x is 3D, and the squeeze operation will work:
z = x.squeeze(1)


     




