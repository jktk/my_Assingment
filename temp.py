import concurrent.futures as c

li = [(1 ,3 )  ]

def add2(a  ,b  ):
    return a**2  + b**2 

with c.ProcessPoolExecutor() as e:
    
    for i in range(50):
        for j in range (50):
            e.submit(add2 , i ,  j )
    