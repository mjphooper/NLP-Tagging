# Natural Language Processing Coursework

 Entity Tagging and Morphology Construction implemented in Python using the NLTK library.
 

## Entity Tagging

 - Train a POS tagger using the Brown corpus.
 - Iterate over and read every email in a directory of untagged emails.
 - For each email, add speaker, location and time tags where relevant by editing the text file.
 - Do this through a combination of Noun identification (using the trained POS tagger), regular expressions  and Pythonic pattern identification and string manipulation.
 

## Morphology Construction

 - Attempting to semantically categorise emails within the Computer Science department in a structured way.
 > An email about a Robotics workshop  would be categorised as 'Robotics -> AI -> Computer Science'
 - Use WordNet to generate a hypernym tree for each word in the email.
 - Expand tree until Computer Science is found as a child, or the maximum search depth is reached.
 - If Computer Science is reached, either add email to existing category, or create a new category if a new path to categorisation is found.

