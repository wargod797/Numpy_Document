
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


a = np.array([[1,2,3],
              [4,5,6]])


# In[3]:


a


# In[4]:


a.shape


# In[5]:


b = a*a


# In[6]:


b


# In[7]:


a = np.ones((1,3))


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


x =([[1,2],
   [3,4]])


# In[10]:


import random
random_matrix = [[random.random() for e in range(2)] for e in range(2)]


# In[11]:


random_matrix


# In[12]:


c = np.matmul(x,random_matrix)


# In[13]:


c


# In[14]:


z = np.array([[5,6],[7,8]])


# In[15]:


z


# In[16]:


z.shape


# In[17]:


k = np.dot(x,z)


# In[18]:


k


# In[19]:


x


# In[20]:


n = np.array([10,20])


# In[21]:


n = n + [5]


# In[22]:


n**2


# In[23]:


np.sqrt(a)


# In[24]:


a = np.array([1, 2, 3]) 
print(a)


# In[25]:


a = np.array([1, 2, 3], dtype = 'int8') 
print(a)


# In[26]:


a


# In[27]:


import numpy as np 
dt = np.dtype([('age',np.int32)]) 
print(dt) 


# In[28]:


import numpy as np 
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
print(student)


# In[29]:


student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student) 
print(a)


# In[30]:


student


# 'b' − boolean
# 
# 'i' − (signed) integer
# 
# 'u' − unsigned integer
# 
# 'f' − floating-point
# 
# 'c' − complex-floating point
# 
# 'm' − timedelta
# 
# 'M' − datetime
# 
# 'O' − (Python) objects
# 
# 'S', 'a' − (byte-)string
# 
# 'U' − Unicode
# 
# 'V' − raw data (void)

# In[31]:


a = np.array([[1,2,3],[4,5,6]]) 
a.shape = (3,2) 
print(a)


# In[32]:


a = np.array([[1,2,3],[4,5,6]]) 
b = a.reshape(3,2) 
print(b)


# In[33]:


k = np.ones((3,3),dtype='int8')


# In[34]:


k


# In[35]:


l = np.zeros((3,3,3,3,3),dtype='int8')


# In[36]:


l


# In[37]:


l.ndim


# In[38]:


l.itemsize


# In[39]:


a.itemsize


# In[40]:


a


# 1	
# C_CONTIGUOUS (C)
# 
# The data is in a single, C-style contiguous segment
# 
# 2	
# F_CONTIGUOUS (F)
# 
# The data is in a single, Fortran-style contiguous segment
# 
# 3	
# OWNDATA (O)
# 
# The array owns the memory it uses or borrows it from another object
# 
# 4	
# WRITEABLE (W)
# 
# The data area can be written to. Setting this to False locks the data, making it read-only
# 
# 5	
# ALIGNED (A)
# 
# The data and all elements are aligned appropriately for the hardware
# 
# 6	
# UPDATEIFCOPY (U)
# 
# This array is a copy of some other array. When this array is deallocated, the base array will be updated with the contents of this array

# In[41]:


a.flags


# 1	
# a
# 
# Input data in any form such as list, list of tuples, tuples, tuple of tuples or tuple of lists
# 
# 2	
# dtype
# 
# By default, the data type of input data is applied to the resultant ndarray
# 
# 3	
# order
# 
# C (row major) or F (column major). C is default

# In[42]:


# convert list to ndarray 
import numpy as np 

x = [1,2,3] 
a = np.asarray(x) 
print(a)


# In[43]:


m = (1,2,3)


# In[44]:


m = np.asarray(m)


# In[45]:


m


# In[46]:


m = list(m)


# In[47]:


m


# In[48]:


m.append(4)


# In[49]:


m


# In[50]:


m = tuple(m)


# In[51]:


m


# In[52]:


z = [(1,2),(3,4)]


# In[53]:


j = np.asarray(z)


# In[54]:


j


# Syntax:
# numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0)
# 
# 
# 1	
# buffer
# 
# Any object that exposes buffer interface
# 
# 2	
# dtype
# 
# Data type of returned ndarray. Defaults to float
# 
# 3	
# count
# 
# The number of items to read, default -1 means all data
# 
# 4	
# offset
# 
# The starting position to read from. Default is 0
# 

# In[55]:


s = b'hello world'
np.frombuffer(s, dtype='S1', count=5, offset=6)


# Syntax: numpy.arange(start, stop, step, dtype)
# 
# 
# 1	
# start
# 
# The start of an interval. If omitted, defaults to 0
# 
# 2	
# stop
# 
# The end of an interval (not including this number)
# 
# 3	
# step
# 
# Spacing between values, default is 1
# 
# 4	
# dtype
# 
# Data type of resulting ndarray. If not given, data type of input is used

# In[56]:


np.arange(10,50,3)


# numpy.linspace(start, stop, num, endpoint, retstep, dtype)

# 1	
# start
# 
# The starting value of the sequence
# 
# 2	
# stop
# 
# The end value of the sequence, included in the sequence if endpoint set to true
# 
# 3	
# num
# 
# The number of evenly spaced samples to be generated. Default is 50
# 
# 4	
# endpoint
# 
# True by default, hence the stop value is included in the sequence. If false, it is not included
# 
# 5	
# retstep
# 
# If true, returns samples and step between the consecutive numbers
# 
# 6	
# dtype
# 
# Data type of output ndarray

# In[57]:


x = np.linspace(10,50,25) 
print(x)


# In[58]:


x.mean()


# In[59]:


np.median(x)


# numpy.logspace(start, stop, num, endpoint, base, dtype)
# 
# 
# 1	
# start
# 
# The starting point of the sequence is basestart
# 
# 2	
# stop
# 
# The final value of sequence is basestop
# 
# 3	
# num
# 
# The number of values between the range. Default is 50
# 
# 4	
# endpoint
# 
# If true, stop is the last value in the range
# 
# 5	
# base
# 
# Base of log space, default is 10
# 
# 6	
# dtype
# 
# Data type of output array. If not given, it depends upon other input arguments

# In[60]:


a = np.logspace(1.0, 5.0, num = 20) 
print(a)


# In[61]:


np.size(a)


# In[62]:


a = np.logspace(1,10,num = 10, base = 2) 
print(a)


# In[63]:


a.shape = (2,5) 


# In[64]:


a


# In[65]:


a = np.arange(10)


# In[66]:


a


# In[67]:


s = slice(2,7,1)


# In[68]:


s


# In[69]:


print(a[s])


# In[70]:


s = slice(2,7,2)


# In[71]:


print(a[s])


# In[72]:


'''import re
n=0
a = []
c=108
l,m,n,o,p = input().split()
while c<113:
    s = str(chr(c))
    a.append((re.sub(r'[^\w]',' ',s)))
    c=c+1
print(a)'''
    


# In[73]:


chr(10)


# In[74]:


ord('\n')


# In[75]:


a = np.arange(10) 
print(a[2:])


# In[76]:


print(a[2:5])


# In[77]:


a = np.array([[1,2,3],[3,4,5],[4,5,6]],dtype='int8')
print(a)


# In[78]:


print(a[0:])


# In[79]:


print(a[1:])


# In[80]:


print(a[2:])


# In[81]:


print(a[:1])


# In[82]:


print(a[:2])


# In[83]:


print(a[:3])


# In[84]:


print(a[1:3])


# In[85]:


print(a[  [0,1]  ])


# In[86]:


print(a[0,1])


# In[87]:


print(a[  [0,2,1]  ])


# In[88]:


a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print ('Our array is:') 
print(a) 

print('The items in the second column are: ')  
print( a[...,1] )  

# Now we will slice all items from the second row 
print('The items in the second row are:')
print( a[1,...] )


# Now we will slice all items from column 1 onwards 
print( 'The items column 1 onwards are:') 
print( a[...,1:])


# In[89]:


x = np.array([[1, 2], [3, 4], [5, 6]]) 
y = x[[0,1,2], [0,1,0]] 
print(y)


# In[90]:


x[[0,1,2],[0,1,0]]


# In[91]:


x[[0,1,2],[0]]


# In[92]:


x[[0,1,2],[1]]


# In[93]:


x[2]


# In[94]:


x[[0,1,2]]


# In[95]:


x[0,1]


# In[96]:


x[0:1]


# In[97]:


x[:1]


# In[98]:


x[:2]


# In[99]:


x[:3]


# In[100]:


x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
print('array',x,end='\n\n\n')
rows = np.array([[0,0],[3,3]])
cols = np.array([[0,2],[0,2]]) 
y = x[rows,cols] 
print(y)


# In[101]:


x[[3,3]]


# In[102]:


x[[0,0]]


# In[103]:


x[[0,2]]


# In[104]:


cols|rows


# In[105]:


# slicing 
z = x[1:4,1:3] 


# In[106]:


z


# In[107]:


x[1:2,1:]


# In[108]:


x[1:,1:]


# In[109]:


x[1:1,1:]


# In[110]:


a = np.array([1, 2+6j, 5, 3.5+5j]) 
print(a[np.iscomplex(a)])


# In[111]:


#Boardcasting
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print(c)


# In[112]:


a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  


# In[113]:


a,b


# In[114]:


print(a+b)


# In[115]:


print(a*b)


# In[116]:


a = np.arange(0,60,5) 
a = a.reshape(3,4) 


# In[117]:


a


# In[118]:


#Transpose
b = a.T 


# In[119]:


b


# In[120]:


print('Sorted in C-style order:')
c = b.copy(order = 'C')


# In[121]:


c


# In[122]:


for x in np.nditer(c):
    print(x)


# In[123]:


#F Order
print(b.copy(order = 'F'))


# In[124]:


c


# In[125]:


#It is possible to force nditer object to use a specific order by explicitly mentioning it.

for x in np.nditer(a, order = 'C'): 
     print(x)


# In[126]:


for x in np.nditer(a, order = 'F'): 
     print(x)


# In[127]:


a = np.arange(0,60,5)
a = a.reshape(3,4)


# In[128]:


a


# In[129]:


for x in np.nditer(a, op_flags = ['readwrite']):
     x[...] = 2*x
print('Modified array is:')
print( a)


# External Loop
# The nditer class constructor has a ‘flags’ parameter, which can take the following values −
# 
# Sr.No.	Parameter & Description
# 1	
# c_index
# 
# C_order index can be tracked
# 
# 2	
# f_index
# 
# Fortran_order index is tracked
# 
# 3	
# multi-index
# 
# Type of indexes with one per iteration can be tracked
# 
# 4	
# external_loop
# 
# Causes values given to be one-dimensional arrays with multiple values instead of zero-dimensional array

# In[130]:


a = np.arange(0,60,5) 
a = a.reshape(3,4) 
for x in np.nditer(a, flags = ['external_loop'], order = 'F'):
            print( x)


# In[131]:


b = np.array([1, 2, 3, 4], dtype = int)


# In[132]:


b


# In[133]:


for x,y in np.nditer([a,b]): 
     print( "%d:%d" % (x,y))


# Transpose Operations
# Sr.No.	Operation & Description
# 
# 1	transpose
# Permutes the dimensions of an array
# 
# 2	ndarray.T
# Same as self.transpose()
# 
# 3	rollaxis
# Rolls the specified axis backwards
# 
# 4	swapaxes
# Interchanges the two axes of an array

# Changing Dimensions
# Sr.No.	Dimension & Description
# 
# 1	broadcast
# Produces an object that mimics broadcasting
# 
# 2	broadcast_to
# Broadcasts an array to a new shape
# 
# 3	expand_dims
# Expands the shape of an array
# 
# 4	squeeze
# Removes single-dimensional entries from the shape of an array

# Joining Arrays
# Sr.No.	Array & Description
# 
# 1	concatenate
# Joins a sequence of arrays along an existing axis
# 
# 2	stack
# Joins a sequence of arrays along a new axis
# 
# 3	hstack
# Stacks arrays in sequence horizontally (column wise)
# 
# 4	vstack
# Stacks arrays in sequence vertically (row wis

# Splitting Arrays
# Sr.No.	Array & Description
# 
# 1	split
# Splits an array into multiple sub-arrays
# 
# 2	hsplit
# Splits an array into multiple sub-arrays horizontally (column-wise)
# 
# 3	vsplit
# Splits an array into multiple sub-arrays vertically (row-wise)

# Adding / Removing Elements
# Sr.No.	Element & Description
# 
# 1	resize
# Returns a new array with the specified shape
# 
# 2	append
# Appends the values to the end of an array
# 
# 3	insert
# Inserts the values along the given axis before the given indices
# 
# 4	delete
# Returns a new array with sub-arrays along an axis deleted
# 
# 5	unique
# Finds the unique elements of an array

# Adding / Removing Elements
# Sr.No.	Element & Description
# 
# 1	resize
# Returns a new array with the specified shape
# 
# 2	append
# Appends the values to the end of an array
# 
# 3	insert
# Inserts the values along the given axis before the given indices
# 
# 4	delete
# Returns a new array with sub-arrays along an axis deleted
# 
# 5	unique
# Finds the unique elements of an array

# Sr.No.	Function & Description
# 
# 1	add()
# Returns element-wise string concatenation for two arrays of str or Unicode
# 
# 2	multiply()
# Returns the string with multiple concatenation, element-wise
# 
# 3	center()
# Returns a copy of the given string with elements centered in a string of specified length
# 
# 4	capitalize()
# Returns a copy of the string with only the first character capitalized
# 
# 5	title()
# Returns the element-wise title cased version of the string or unicode
# 
# 6	lower()
# Returns an array with the elements converted to lowercase
# 
# 7	upper()
# Returns an array with the elements converted to uppercase
# 
# 8	split()
# Returns a list of the words in the string, using separatordelimiter
# 
# 9	splitlines()
# Returns a list of the lines in the element, breaking at the line boundaries
# 
# 10	strip()
# Returns a copy with the leading and trailing characters removed
# 
# 11	join()
# Returns a string which is the concatenation of the strings in the sequence
# 
# 12	replace()
# Returns a copy of the string with all occurrences of substring replaced by the new string
# 
# 13	decode()
# Calls str.decode element-wise
# 
# 14	encode()
# Calls str.encode element-wise

# In[134]:


print( 'Bitwise AND of 13 and 17:') 
print( np.bitwise_and(13, 17))


# BITWISE OPERATION
# 
# Sr.No.	Operation & Description
# 
# 1	bitwise_and
# Computes bitwise AND operation of array elements
# 
# 2	bitwise_or
# Computes bitwise OR operation of array elements
# 
# 3	invert
# Computes bitwise NOT
# 
# 4	left_shift
# Shifts bits of a binary representation to the left
# 
# 5	right_shift
# Shifts bits of binary representation to the right

# STRING OPERATION
# 
# Sr.No.	Function & Description
# 
# 1	add()
# Returns element-wise string concatenation for two arrays of str or Unicode
# 
# 2	multiply()
# Returns the string with multiple concatenation, element-wise
# 
# 3	center()
# Returns a copy of the given string with elements centered in a string of specified length
# 
# 4	capitalize()
# Returns a copy of the string with only the first character capitalized
# 
# 5	title()
# Returns the element-wise title cased version of the string or unicode
# 
# 6	lower()
# Returns an array with the elements converted to lowercase
# 
# 7	upper()
# Returns an array with the elements converted to uppercase
# 
# 8	split()
# Returns a list of the words in the string, using separatordelimiter
# 
# 9	splitlines()
# Returns a list of the lines in the element, breaking at the line boundaries
# 
# 10	strip()
# Returns a copy with the leading and trailing characters removed
# 
# 11	join()
# Returns a string which is the concatenation of the strings in the sequence
# 
# 12	replace()
# Returns a copy of the string with all occurrences of substring replaced by the new string
# 
# 13	decode()
# Calls str.decode element-wise
# 
# 14	encode()
# Calls str.encode element-wise

# In[135]:


a = np.array([0,30,45,60,90]) 

print( 'Sine of different angles:') 
# Convert to radians by multiplying with pi/180 
print(np.sin(a*np.pi/180)) 
print( '\n')  

print( 'Cosine values for angles in array:') 
print(np.cos(a*np.pi/180) )
print( '\n')  

print('Tangent values for given angles:')
print(np.tan(a*np.pi/180) )


# In[136]:


a = np.array([0,30,45,60,90])


# In[137]:


a


# In[138]:


np.sin(a*np.pi/180)


# In[141]:


#'Compute sine inverse of angles. Returned values are in radians.'
sin = np.sin(a*np.pi/180)
np.arcsin(sin) 


# In[145]:


print('Check result by converting to degrees:' )
inv = np.arcsin(sin) 
print(np.degrees(inv))


# In[147]:


cos =( np.cos(a*np.pi/180)) 
print( cos) 


# In[148]:


#inverse of cos is sec
print( 'Inverse of cos:' )
inv =( np.arccos(cos)) 
print( inv)


# In[149]:


print( 'In degrees:') 
print( np.degrees(inv)) 


# In[150]:


np.tan(a*np.pi/180) 


# In[151]:


a = np.arange(9, dtype = np.float_).reshape(3,3) 


# In[152]:


a


# In[154]:


np.cos(a*np.pi/180)


# In[157]:


#Array Operations
a = np.arange(9, dtype = np.float_).reshape(3,3) 
b = np.array([10,10,10])


# In[158]:


a, b


# In[159]:


print( 'Add the two arrays:') 
print( np.add(a,b)) 


# In[160]:


print( 'Subtract the two arrays:') 
print( np.subtract(a,b))


# In[161]:


print( 'Multiply the two arrays:') 
print( np.multiply(a,b))


# In[162]:


print( 'Divide the two arrays:') 
print( np.divide(a,b))


# In[164]:


print( 'Matrix multiplication of two arrays the two arrays:') 
print( np.matmul(a,b))


# numpy.reciprocal()
# This function returns the reciprocal of argument, element-wise. For elements with 
# absolute values larger than 1, the result is always 0 because of the way in which Python handles integer division. 
# For integer 0, an overflow warning is issued.

# In[165]:


a = np.array([0.25, 1.33, 1, 0, 100])
print(a)


# In[166]:


print( np.reciprocal(a)) 


# In[181]:


print(np.reciprocal(a))


# In[182]:


b = np.array([100], dtype = float) 


# In[183]:


b


# In[184]:


print(np.reciprocal(b))


# In[185]:


a = np.array([10,100,1000]) 


# In[186]:


a


# In[188]:


#Return Power of the Values
print( 'Applying power function:' )
print( np.power(a,2)) 


# In[189]:


b =( np.array([1,2,3])) 
print( b) 


# In[190]:


np.power(a,b)


# In[191]:


print( 'Applying power function again:') 
print( np.power(a,b))


# numpy.mod()
# 
# This function returns the remainder of division of the corresponding elements 
# in the input array. The function numpy.remainder() also produces the same result.

# In[193]:


a = np.array([10,20,30]) 
b = np.array([3,5,7])


# In[194]:


a,b


# In[195]:


print( np.mod(a,b)) 


# In[196]:


print(np.remainder(a,b))


# The following functions are used to perform operations on array with complex numbers.
# 
# numpy.real() − returns the real part of the complex data type argument.
# 
# numpy.imag() − returns the imaginary part of the complex data type argument.
# 
# numpy.conj() − returns the complex conjugate, which is obtained by changing the sign of the imaginary part.
# 
# numpy.angle() − returns the angle of the complex argument. The function has degree parameter. If true, the angle in the degree is returned, otherwise the angle is in radians.

# In[197]:


a = np.array([-5.6j, 0.2j, 11. , 1+1j]) 


# In[198]:


a


# In[199]:


np.real(a)


# In[200]:


np.imag(a)


# In[201]:


#Inverse the sign of complex numbers
np.conj(a) 


# In[202]:


np.angle(a)


# In[203]:


np.angle(a, deg = True)


# In[205]:


#min and max of numpy don't work on complex numbers
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 
np.amin(a,1) 


# In[208]:


#print minimum values of rows
np.amin(a,1)


# In[209]:


#print the minium values of coloumns
np.amin(a,0)


# In[210]:


#Returns the Maximum Values of the array
np.amax(a)


# In[212]:


#return the maximum values of the coloumns
np.amax(a, axis=0)


# In[213]:


#return the maximum values of the rows
np.amax(a, axis=0)


# numpy.ptp()
# 
# 
# The numpy.ptp() function returns the range (maximum-minimum) of values along an axis.

# In[214]:


a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 


# In[215]:


# returns maximum-mininum in array(a)
np.ptp(a)


# In[216]:


# returns maximum-mininum in row
np.ptp(a, axis = 1) 


# In[218]:


# returns maximum-mininum in columns
np.ptp(a, axis = 0)


# numpy.percentile()
# 
# 
# Percentile (or a centile) is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations fall. The function numpy.percentile() takes the following arguments.
# 
# 
# Sr.No.	Argument & Description
# 
# 1	a
# 
# Input array
# 
# 2	q
# 
# The percentile to compute must be between 0-100
# 
# 3	axis
# 
# The axis along which the percentile is to be calculated

# In[219]:


a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 


# In[228]:


np.percentile(a,50)


# In[229]:


np.percentile(a,30)


# In[230]:


np.percentile(a,50, axis = 1) 


# In[231]:


np.percentile(a,50, axis = 0) 


# In[232]:


np.percentile(a,50, axis = 0)


# numpy.median()
# 
# 
# Median is defined as the value separating the higher half of a data sample from the lower half. The numpy.median() function is used as shown in the following program.

# In[233]:


np.median(a)


# In[235]:


a


# In[234]:


np.median(a, axis = 0)


# In[236]:


np.median(a, axis = 1)


# numpy.mean()
# 
# 
# Arithmetic mean is the sum of elements along an axis divided by the number of elements. The numpy.mean() function returns the arithmetic mean of elements in the array. If the axis is mentioned, it is calculated along it.

# In[237]:


a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 


# In[238]:


np.mean(a)


# In[239]:


np.mean(a, axis = 0)


# In[240]:


np.mean(a, axis = 1)


# numpy.average()
# 
# 
# Weighted average is an average resulting from the multiplication of each component by a factor reflecting its importance. The numpy.average() function computes the weighted average of elements in an array according to their respective weight given in another array. The function can have an axis parameter. If the axis is not specified, the array is flattened.
# 
# Considering an array [1,2,3,4] and corresponding weights [4,3,2,1], the weighted average is calculated by adding the product of the corresponding elements and dividing the sum by the sum of weights.
# 
# Weighted average = (1*4+2*3+3*2+4*1)/(4+3+2+1)

# In[242]:


np.average(a, axis = 1)


# In[243]:


np.average(a, axis = 0)


# In[244]:


np.average(a)


# Standard Deviation
# 
# 
# Standard deviation is the square root of the average of squared deviations from mean. The formula for standard deviation is as follows −

# std = sqrt(mean(abs(x - x.mean())**2))

# In[250]:


np.std([10,20,30,40,50])


# Variance
# Variance is the average of squared deviations, i.e., mean(abs(x - x.mean())**2). In other words, the standard deviation is the square root of variance.

# In[259]:


np.var([1,2,3,4])


# In[261]:


a=np.array([[3,7],[9,1]]) 


# In[262]:



np.sort(a)


# In[263]:


np.sort(a, axis = 0) 


# In[264]:


np.sort(a, axis = 1) 


# In[267]:


a.itemsize


# numpy.argsort()
# 
# 
# The numpy.argsort() function performs an indirect sort on input array, along the given axis and using a specified kind of sort to return the array of indices of data. This indices array is used to construct the sorted array.

# In[279]:


x = np.array([3, 1, 2]) 


# In[280]:


x


# In[284]:


y= np.argsort(x)
y


# In[283]:


#get the sorted values use this
x[y] 


# In[ ]:


#'Index of maximum number in flattened array' 


# In[291]:


a.flatten()


# numpy.lexsort()
# 
# function performs an indirect sort using a sequence of keys. The keys can be seen as a column in a spreadsheet. The function returns an array of indices, using which the sorted data can be obtained. Note, that the last key happens to be the primary key of sort.

# In[285]:


nm = ('raju','anil','ravi','amar') 
dv = ('f.y.', 's.y.', 's.y.', 'f.y.') 
ind = np.lexsort((dv,nm)) 


# In[286]:


ind


# In[288]:


np.argmax(a) 


# In[290]:


np.argmin(a) 


# In[287]:


#'Use this index to get sorted data:' 

[nm[i] + ", " + dv[i] for i in ind] 


# numpy.nonzero()
# The numpy.nonzero() function returns the indices of non-zero elements in the input array.

# In[298]:


a = np.array([[30,40,0],[0,20,10],[50,0,60]]) 


# In[299]:


print( a )


# In[308]:


print( np.nonzero (a))


# In[302]:


y = np.where(x > 3) 


# In[303]:


print(y)


# In[304]:


print(a[y])


# In[305]:


x = np.arange(9.).reshape(3, 3) 


# In[306]:


x


# In[309]:


y = np.where(x > 3) 


# In[310]:


y


# In[311]:


#Use these indices to get elements satisfying the condition' 
x[y]


# numpy.extract()
# 
# 
# The extract() function returns the elements satisfying any condition

# In[312]:


x = np.arange(9.).reshape(3, 3) 


# In[313]:


condition = np.mod(x,2) == 0 


# In[314]:


condition


# In[315]:


np.extract(condition, x)


# numpy.ndarray.byteswap()
# 
# The numpy.ndarray.byteswap() function toggles between the two representations: bigendian and little-endian.

# In[338]:


a = np.array([1, 256, 8755], dtype = np.int16) 


# In[339]:


a


# No Copy
# 
# Simple assignments do not make the copy of array object. Instead, it uses the same id() of the original array to access it. The id() returns a universal identifier of Python object, similar to the pointer in C.
# 
# Furthermore, any changes in either gets reflected in the other. For example, the changing shape of one will change the shape of the other too.

# In[342]:


a = np.arange(6)
a


# In[341]:


id(a)


# In[343]:


b = a 


# In[344]:


id(b)


# In[345]:


b.shape = (3,2)


# In[346]:


a


# In[348]:


b
#if b cahnges a also changes


# View or Shallow Copy
# 
# NumPy has ndarray.view() method which is a new array object that looks at the same data of the original array. Unlike the earlier case, change in dimensions of the new array doesn’t change dimensions of the original.

# In[349]:


a = np.arange(6).reshape(3,2) 


# In[350]:


a


# In[351]:


b = a.view() 


# In[352]:


b


# In[353]:


id(b)


# In[356]:


b.shape = (2,3)


# In[357]:


b


# In[358]:


a
#now you can see if u change value of b, a value doesn't change


# In[359]:


a = np.array([[10,10], [2,3], [4,5]]) 


# In[360]:


a


# In[361]:


s = a[:, :2] 


# In[362]:


s


# Deep Copy
# 
# The ndarray.copy() function creates a deep copy. It is a complete copy of the array and its data, and doesn’t share with the original array.

# In[367]:


a = np.array([[10,10], [2,3], [4,5]]) 
a


# In[364]:


b = a.copy() 


# In[365]:


b


# In[366]:


a


# In[369]:


#b does not share any memory of a 
print( b is a )


# In[370]:


b[0,0] = 100 


# In[371]:


b,a


# In[372]:


#the values doesn't changed


# NumPy package contains a Matrix library numpy.matlib. This module has functions that return matrices instead of ndarray objects.
# 
# matlib.empty()
# The matlib.empty() function returns a new matrix without initializing the entries. The function takes the following parameters.
# 
# numpy.matlib.empty(shape, dtype, order)
# 
# 
# 
# Where,
# 
# Sr.No.	Parameter & Description
# 1	shape
# 
# int or tuple of int defining the shape of the new matrix
# 
# 2	Dtype
# 
# Optional. Data type of the output
# 
# 3	order
# 
# C or F

# In[373]:


import numpy.matlib


# In[374]:


print( np.matlib.empty((2,2))) 


# numpy.matlib.zeros()
# This function returns the matrix filled with zero

# In[375]:


np.matlib.zeros((2,2))


# In[376]:


np.matlib.ones((2,2))


# numpy.matlib.eye()
# 
# This function returns a matrix with 1 along the diagonal elements and the zeros elsewhere. The function takes the following parameters.

# numpy.matlib.eye(n, M,k, dtype)
# 
# 
# Where,
# 
# Sr.No.	Parameter & Description
# 1	n   The number of rows in the resulting matrix
# 
# 2	M
# The number of columns, defaults to n
# 
# 3	k
# Index of diagonal
# 
# 4	dtype
# Data type of the output

# In[378]:


np.matlib.eye(n = 3, M = 4, k = 0, dtype = float)


# numpy.matlib.identity()
# 
# 
# The numpy.matlib.identity() function returns the Identity matrix of the given size. An identity matrix is a square matrix with all diagonal elements as 1.

# In[377]:


np.matlib.identity(5, dtype = float)


# In[379]:


np.matlib.rand(3,3)


# In[380]:


np.matrix('1,2;3,4') 


# In[383]:


np.asmatrix (j)


# NumPy package contains numpy.linalg module that provides all the functionality required for linear algebra. Some of the important functions in this module are described in the following table.
# 
# Sr.No.	Function & Description
# 1	dot
# Dot product of the two arrays
# 
# 2	vdot
# Dot product of the two vectors
# 
# 3	inner
# Inner product of the two arrays
# 
# 4	matmul
# Matrix product of the two arrays
# 
# 5	determinant
# Computes the determinant of the array
# 
# 6	solve
# Solves the linear matrix equation
# 
# 7	inv
# Finds the multiplicative inverse of the matrix

# In[384]:


from matplotlib import pyplot as plt 


# In[385]:


x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y) 
plt.show()


# Sr.No.	Character & Description
# 1	
# '-'
# 
# Solid line style
# 
# 2	
# '--'
# 
# Dashed line style
# 
# 3	
# '-.'
# 
# Dash-dot line style
# 
# 4	
# ':'
# 
# Dotted line style
# 
# 5	
# '.'
# 
# Point marker
# 
# 6	
# ','
# 
# Pixel marker
# 
# 7	
# 'o'
# 
# Circle marker
# 
# 8	
# 'v'
# 
# Triangle_down marker
# 
# 9	
# '^'
# 
# Triangle_up marker
# 
# 10	
# '<'
# 
# Triangle_left marker
# 
# 11	
# '>'
# 
# Triangle_right marker
# 
# 12	
# '1'
# 
# Tri_down marker
# 
# 13	
# '2'
# 
# Tri_up marker
# 
# 14	
# '3'
# 
# Tri_left marker
# 
# 15	
# '4'
# 
# Tri_right marker
# 
# 16	
# 's'
# 
# Square marker
# 
# 17	
# 'p'
# 
# Pentagon marker
# 
# 18	
# '*'
# 
# Star marker
# 
# 19	
# 'h'
# 
# Hexagon1 marker
# 
# 20	
# 'H'
# 
# Hexagon2 marker
# 
# 21	
# '+'
# 
# Plus marker
# 
# 22	
# 'x'
# 
# X marker
# 
# 23	
# 'D'
# 
# Diamond marker
# 
# 24	
# 'd'
# 
# Thin_diamond marker
# 
# 25	
# '|'
# 
# Vline marker
# 
# 26	
# '_'
# 
# Hline marker
# 
# The following color abbreviations are also defined.
# 
# Character	Color
# 'b'	Blue
# 'g'	Green
# 'r'	Red
# 'c'	Cyan
# 'm'	Magenta
# 'y'	Yellow
# 'k'	Black
# 'w'	Whi

# In[386]:


x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y,"ob") 
plt.show() 


# In[387]:


# Compute the x and y coordinates for points on a sine curve 
x = np.arange(0, 3 * np.pi, 0.1) 
y = np.sin(x) 
plt.title("sine wave form") 

# Plot the points using matplotlib 
plt.plot(x, y) 
plt.show()


# In[388]:


x = np.arange(0, 3 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
plt.subplot(2, 1, 1)
   
# Make the first plot 
plt.plot(x, y_sin) 
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 1, 2) 
plt.plot(x, y_cos) 
plt.title('Cosine')  
   
# Show the figure. 
plt.show()


# In[389]:


y


# In[390]:


from matplotlib import pyplot as plt 
x = [5,8,10] 
y = [12,16,6]  

x2 = [6,9,11] 
y2 = [6,15,7] 
plt.bar(x, y, align = 'center') 
plt.bar(x2, y2, color = 'g', align = 'center') 
plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis')  

plt.show()


# In[391]:


a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
np.histogram(a,bins = [0,20,40,60,80,100]) 
hist,bins = np.histogram(a,bins = [0,20,40,60,80,100]) 


# In[392]:


a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
plt.hist(a, bins = [0,20,40,60,80,100]) 
plt.title("histogram") 
plt.show()


# In[394]:


#Save File
a = np.array([1,2,3,4,5]) 
np.save('outfile',a)


# In[ ]:


#Save Text

