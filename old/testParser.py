
#-------------------------------
# regression based on the level from 2-5 you chosed
#-------------------------------
import pandas as pd
PATH = "/Users/mila/Documents/GitHub/boron-main/data/2022-11-28 B Carb Sy MC JX AG splitstream_20221128-203622/001_A.exp"
df = pd.read_csv(PATH, sep='\t', header=22)  # Hi Jie, I already inserted the additional argument here
print(df)


with open(PATH, "r") as openfile:
    content = openfile.read()

print(content)

_start = content.find("Cycle\tTime")
print(_start)

_end = content.find("***\tCup")
print(_end)

myTable = content[_start:_end-1]
print(myTable)
