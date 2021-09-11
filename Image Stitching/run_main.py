import os




def getChunks(path , n):

      for i in range(0 , len(path) , n):
            yield path[i : i + n]

root = 'rgb_data'
path = [os.path.join(root , f) for f in os.listdir(root) if not f.endswith('.py') and not f.endswith('.db') and not f.startswith('mask') and not f.endswith('.png')]

print(path , len(path))

k = 3

dirs = ['left' , 'center' , 'right']

idx = 0
for ctr , data in enumerate(getChunks(path , k)):

      args = '1' + ' ' + data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + dirs[ctr]

      print(args)
      os.system('python main.py' + ' ' + args)
      

      
