import os 
def data(path="dataset"):  
    if not os.path.exists(path): 
        print("data directory not found:", path) 
        return data 
    x = ["jack", "queen", "king"]   
    ctg = ["withMask", "withoutMask"] 
    total = 0     
    print("checking path:", os.path.abspath(path))
    print()     
    for folder in x:  
        print("Section:", folder)
        for cat in ctg:   
            path2 = os.path.join(path, folder, cat)   
            if os.path.exists(path2):
                list2 = os.listdir(path2)
                files = len(list2)     
                total += files   
                print(" ", cat, "->", files)
            else:   
             print(" ", cat, "-> file missing:", path2)
        print()
    print("total images found:", total)     
if  __name__ == "imp":
    data()      