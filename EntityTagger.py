'''Entity Tagging
Matthew Hooper
1790569
'''

#------------------------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------------------------

import nltk
import pickle
import math
import re
from nltk.corpus import brown
from nltk.corpus import words
from nltk.tag import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
import os

#------------------------------------------------------------------------------------------------
# Get the names files
#------------------------------------------------------------------------------------------------

ext = ['family','male','female']
storedNames = {}
for i in range(3):
    storedNames[ext[i]] = (open('Data/Names/names.'+ext[i]).read().split("\n"))


#------------------------------------------------------------------------------------------------
# Helpful functions
#------------------------------------------------------------------------------------------------

def getTextSegment(text,index,substring): #Returns the string between the index and next occurence of substring
    text = text[index:]
    endIndex = text.find(substring)
    return text[:endIndex]

def insertString(text,string,index):
    return text[:index] + string + text[index:]

def getGreater(a,b):
    if a >= b: return a
    else: return b

#Function to check if any string in the list is a substring of the given string.
def matchList(str_ls,str,matchCase):
    for item in str_ls:
        if matchCase: item = item.lower() ; str = str.lower()
        if item in str:
            return True
    return False


#------------------------------------------------------------------------------------------------
# Arrange and train the tagger
#------------------------------------------------------------------------------------------------

def saveTagger(tagger):
    saveFile = open("backoff.pickle","wb")
    pickle.dump(tagger, saveFile)
    saveFile.close()
def loadTagger():
    readFile = open("backoff.pickle", "rb")
    tagger = pickle.load(readFile)
    readFile.close()
    return tagger

# Split the training data and train.
brownTagged = brown.tagged_sents(tagset='universal')
trainSplit = math.floor(0.75 * len(brownTagged))

train_sents = brownTagged[:trainSplit]
test_sents = brownTagged[trainSplit:]


def backoffTagger(train_sents,tagger_classes,backoff=None,cutoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents,backoff=backoff,cutoff=cutoff)
    return backoff
#tagger = backoffTagger(train_sents,[UnigramTagger,BigramTagger,TrigramTagger],backoff=DefaultTagger('NOUN'),cutoff=1) #Use UBT combination for best results.
#saveTagger(tagger)
tagger = loadTagger()


#------------------------------------------------------------------------------------------------
# Email class
#------------------------------------------------------------------------------------------------

class Email():
    def __init__(self,name,email,tagger):
        self.name = name
        self.email = email
        self.tagger = tagger

        self.seperateAbstract()
        self.categoriseNouns_loadData()


    def seperateAbstract(self):
        startIndex = self.email.lower().find('abstract:')
        endIndex = startIndex + len('abstract:')

        if startIndex != -1:
            self.header = self.email[:endIndex]
            self.content = self.email[endIndex:]


    # TAG TIMES WITHIN THE TEXT.

    def tagTimes(self,s):
        timeRegEx = r"([0-9]|1[0-2])(:)([0-5][0-9])"
        sArray = s.split(" ")
        s = ""

        dateCount = 0
        ignoredIndexes = [] #To simulate deletion- deleting in for loop will cause indexing errors, and iterating backwards causes a backwards sentence!

        for i in range(len(sArray)):
            if i not in ignoredIndexes:
                if re.match(timeRegEx,sArray[i]): #Current word is a time
                    dateType = "etime>"
                    if dateCount % 2 == 0: dateType = "stime>"


                    if i < len(sArray)-1:
                        if "am" in sArray[i+1].lower()  or "pm" in sArray[i+1].lower() : #If next word is am or pm, include this within the tags.
                            sArray[i] = sArray[i] + " " + sArray[i+1]
                            ignoredIndexes.append(i+1)


                    sArray[i] = ("<"+dateType)+sArray[i]+("</"+dateType)
                    dateCount += 1
                s = s + sArray[i] + " "
        return s

    # header = tagTimes(header), content =...




    # CATEGORISE NOUNS

    def categoriseNouns_loadData(self):
        self.categoriseNouns_data = {
                "nameTitles" : ["Mr","Mrs","Dr","Prof","Professor","Ms"],
                "speakerPrepositions" : ["by","from","with","speaker","lecturer"],
                "locationPrepositions" : ["above","at","in","room","theatre","inside","between","to"],
                "nameExceptions" : ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
        }
    def isName(self,str,listsToCheck):
        str = str.replace(" ","")
        def checkNameLists(str,lss): #Where listsToCheck is a str[]
            if len(lss) > 0:
                return str.lower() in (name.lower() for name in storedNames[lss[-1]]) or self.isName(str,lss[:-1])
        return checkNameLists(str,listsToCheck) or matchList(self.categoriseNouns_data["nameTitles"],str,True)

    def categoriseNouns(self,s):

        #Initialise the varaibles
        sArray = s.split(" ") #Split the array on spaces.
        sLength = len(sArray)
        s = ""

        categories = ["normal"] * sLength

        for position in range(sLength):
            #Only get involved if word is a noun.
            if sArray[position].find("<n>") != -1:

                #Strip <n> tags
                sArray[position] = re.sub("<n>","",sArray[position])
                sArray[position] = re.sub("</n>","",sArray[position])
                word = sArray[position]


                if position == 0:
                    #Check for speaker
                    if self.isName(word,["male","female"]):
                        sArray[position] = "<speaker>"+sArray[position]+"</speaker>"
                        categories[position] = "speaker"
                        continue
                    #V. unlikely for sentence to be started with location!

                prevWord = sArray[position-1] # Allowed to do this now.
                prevPos = position-1

                if word[0].isupper(): #Make sure this is a proper noun if it is occurring mid sentence.
                    #<speaker> TAGS ------------------------------------------------------
                    speakerWeight = 0
                    if prevWord.lower() in self.categoriseNouns_data["speakerPrepositions"]: speakerWeight += 1
                    if prevWord.lower() == "speaker:": speakerWeight += 2
                    if categories[prevPos] == "speaker": speakerWeight += 2
                    if matchList(self.categoriseNouns_data["nameTitles"],prevWord,False): speakerWeight += 2

                    #Cases to cancel the speaker tag
                    if word == "\n" or word.isspace(): speakerWeight = -2
                    if ("where:" in prevWord.lower()) or ("place" in prevWord.lower()): speakerWeight = -2

                    if ((self.isName(word,["male","female","family"]) and speakerWeight >= 0)
                    or (not self.isName(word,["male","female","family"]) and speakerWeight >= 2)):
                        sArray[position] = "<speaker>"+sArray[position]+"</speaker>"
                        categories[position] = "speaker"
                        continue

                    #<location> TAGS ------------------------------------------------------

                    if (categories[prevPos] == "location"
                    or prevWord.lower() in self.categoriseNouns_data["locationPrepositions"]
                    or ("where:" in prevWord.lower()) or ("place" in prevWord.lower())):
                        sArray[position] = "<location>"+sArray[position]+"</location>"
                        categories[position] = "location"

            else: #Doesn't have <n> tags... check if it's a number! Most likely will be a room number.
                if sArray[position].isdigit():
                    sArray[position] = "<location>"+sArray[position]+"</location>"
                    categories[position] = "location"

        #Reconstuct the sentence, and merge parallel tags
        for position in range(sLength):

            if position < sLength-1:
                if categories[position] == categories[position+1]: #Current category matches next category:
                    tagSize = len(categories[position])+2
                    category = categories[position]
                    sArray[position]= sArray[position].replace("</"+category+">","")
                    sArray[position+1]= sArray[position+1].replace("<"+category+">","")

            s = s + sArray[position] + " "

        #Return the reformatted string
        return s




    # FIND NOUNS USING POS TAGGER

    def getNouns(self,s): # S will be a natural string.
        #Split the string on whitespace
        sArray = s.split(" ")
        s = ""
        punctuation = ".,:?!-;()"

        tagged = self.tagger.tag(sArray)
        # We're only interested in Nouns at the moment to find people and places. Strip everything but nouns!
        for pair in tagged:
            if pair[1] == "NOUN":
                word = pair[0]
                #Remove empty strings found
                if word == "":
                    continue
                #Slot puncuation around the tags
                punctuation_prefix = ""
                punctuation_suffix = ""
                if word[0] in punctuation:
                    punctuation_prefix = word[0]
                    word = word[1:]
                if word[len(word)-1] in punctuation:
                    punctuation_suffix = word[len(word)-1]
                    word = word[:len(word)-1]
                s = s + punctuation_prefix + " <n>" + word + "</n>" + punctuation_suffix #Tag the noun termporarily for our own usage.
            else:
                s = s + " " + pair[0]
        return s;


    # FIND THE PARAGRAPH AND SENTENCE TAGS

    def structureTags(self,ls): #If previous sent wasn't a paragr
        ret = '<paragraph> ' #INSERT A \n\n if looking at a non-paragraph directly after a paragraph.
        for i in range(len(ls)):
            ret = ret + '<sentence> ' + str(ls[i]) + ' </sentence> '
        ret = ret + '</paragraph>\n\n'
        return ret



    #RUN IT ALL

    def printToScreen(self):
        print(self.header+self.content)

    def writeToDocument(self,s):
        outputFile = open("Output/"+self.name, "w+")
        outputFile.write(self.header+self.content)
        outputFile.close()

    def run(self):

        #Tag time!
        self.content = self.tagTimes(self.content)
        self.header = self.tagTimes(self.header)

        #Tokenize paragraphs naively
        paragraphs = self.content.split('\n\n')
        self.content = '' #Reset the content to be refilled.

        for i in range(len(paragraphs)):
            paragraphs[i] = nltk.sent_tokenize(paragraphs[i])
            p = paragraphs[i]

            # Pick apart the sentence.
            for j in range(len(p)):
                p[j] = self.getNouns(p[j]) #Put in speaker and location tags.
                p[j] = self.categoriseNouns(p[j])

            # Apply structure tagging according to the paragraph contents.
            if len(p) > 1:
                self.content = self.content + self.structureTags(p) # Valid paragraph with sentences.
            elif len(p) == 1:
                self.content = self.content + p[0] + '\n\n' # One sentence? Probably not a paragraph. Just add.
            else:
                self.content = self.content + '\n\n' #Nothing? Just new line characters.

        self.writeToDocument(self.header+self.content)


#------------------------------------------------------------------------------------------------
# Run on the email set
#------------------------------------------------------------------------------------------------

#'''
dir = 'Data/Emails/untagged'
for filename in os.listdir(dir):
    print(dir+"/"+filename)
    if filename[len(filename)-3:] == "txt":
        with open(dir+"/"+filename,'r') as openedEmail:
            email = openedEmail.read()
            if len(email) > 0:
                currentEmail = Email(filename,email,tagger)
                currentEmail.run()
    else:
        continue
#'''

print("All written successfully!")

'''
with open('Data/Emails/untagged/333.txt','r') as openedEmail:
    email = openedEmail.read()

    currentEmail = Email(email,tagger)
    currentEmail.run()
'''
