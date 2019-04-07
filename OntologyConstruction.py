'''Ontology Construction
Matthew Hooper
1790569
'''

import  nltk
import os
from nltk.corpus import wordnet
from collections import Counter

#Define the class for the tree
class Tree(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def getData(self):
        return self.data

    def getChildren(self):
        return self.children

    def addChild(self, tree):
        self.children.append(tree)


#Create the cateogries list
categories = {
    "computer_science" : [{}]
}
#We are searching for things in the domain category of computer science!

class SynsetSearch: #
    def __init__(self,word):
        self.word = self.formatWord(word)
        self.goal = wordnet.synsets("computer_science")[0]

    def formatWord(self,w):
        return w.replace(" ","_").lower()

    def generateSynsetHypernyms(self,synset):
        hypernyms = []
        for h in synset.hypernyms():
            hypernyms.append(h)
        return hypernyms


    def categorise(self,treeDepth):
        startTree = Tree(wordnet.synsets(self.word)[0])
        result = self.createHypernymTree(startTree,None,treeDepth)

        resultReturn = False

        if result is not None:
            resultReturn = result
        return resultReturn


    def createHypernymTree(self,tree,result,depth):
        #Get the tree contents
        data = tree.data

        #Is the data what we're looking for? #Check if we're one level away from Computer science. if so, we've found our categorisation!
        if self.goal in data.hypernyms():
            result = data
            return result

        #Depth base case
        if depth == 0: return #Don't go above 4. At this point, we're just wasting time.

        #Generate the tree's children
        tree.children = self.generateSynsetHypernyms(data)

        #Recursive step
        for child in tree.children:
            return self.createHypernymTree(Tree(child),result,depth-1)



#------------------------------------------------------------------------------------------------
# Email class  HYPERNYM = LESS DETAIL: METAL FROM CUTLERY , HYPONYM = MORE DETAIL: FORK FROM CUTLERY
#------------------------------------------------------------------------------------------------

class Email():
    def __init__(self,name,email,treeDepth):
        self.name = name
        self.email = email
        self.treeDepth = treeDepth

    def searchWholeEmail(self):
        words = self.email.split(" ")
        for word in words:
            if len(wordnet.synsets(word)) > 0:
                found = (SynsetSearch(word)).categorise(self.treeDepth)
                if found is not False:
                    #If a category was foudn, either create a new category or add it to existing.
                    #Extract just the name
                    name = found.name().split(".")[0]
                    if name not in categories["computer_science"][0]:
                        (categories["computer_science"][0])[name] = [self.name]
                    else:
                        categories["computer_science"][0][name].append(self.name)
                    break
        categories["computer_science"].append(self.name)



    def run(self):
        self.searchWholeEmail()




#------------------------------------------------------------------------------------------------
#Run and load the emails!
#------------------------------------------------------------------------------------------------

'''
name = "312"
with open("Data/Emails/untagged/"+name+".txt",'r') as openedEmail:
    name = "312"
    email = openedEmail.read()
tst = Email(name,email)
tst.run()
'''

#'''
print("Processing...")
dir = 'Data/Emails/untagged'
depth = 1
for filename in os.listdir(dir):
    if filename[len(filename)-3:] == "txt":
        with open(dir+"/"+filename,'r') as openedEmail:
            email = Email(filename,openedEmail.read(),depth)
            email.run()


# Print out results
print("*"*60)
print("Total files processed: "+str(len(os.listdir(dir))))

numberCategorised = 0
for key in categories["computer_science"][0]:
    numberCategorised = numberCategorised + len(categories["computer_science"][0][key])

print("% categorised: "+str((len(categories["computer_science"])-1)/numberCategorised)+" at depth "+str(depth))
