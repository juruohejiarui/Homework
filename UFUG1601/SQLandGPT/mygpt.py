import pprint
import os
import random
import json
def tokenize(text):
  words = text.split(" ")
  return words
def select_top_from_list(sorted_prob_dict, top):
    # find min between top and len(sorted_prob_dict)
    min_val = min(top, len(sorted_prob_dict))
    ran_num = random.randint(0, min_val-1)
    return sorted_prob_dict[ran_num][0]

class GPT:
  db = {}
  GPT_DB_FILE="GPT_DB.json"

  def build(self, sentences):
      # devide this sentences into words
      words = tokenize(sentences)
      for i in range(len(words)-1):
          first_word = words[i]
          second_word = words[i+1]
          
          if first_word in self.db:
              prob_dict = self.db[first_word]
              if second_word in prob_dict:
                  prob_dict[second_word] += 1
              else:
                  prob_dict[second_word] = 1
          else:
              self.db[first_word] = {second_word: 1}
              
  def infer(self, first_word, num_words):
    result = first_word
    for _ in range(num_words):
        if first_word in self.db:
            prob_dict = self.db[first_word]
            ##pprint.pprint(prob_dict)
            # sort the prob_dict by value
            sorted_prob_dict = sorted(prob_dict.items(),
                                    key=lambda x: x[1], reverse=True)
            
            next_word = select_top_from_list(sorted_prob_dict, 3)
            # print(next_word, end=" ")
            result += " " + next_word
            first_word = next_word
            
    return result
        
  def build_gpt_DB_from_file(self, filename):
  # Open file
    with open(filename, "r") as file:
      for line in file:
        # print("training: ", line)
        self.build(line)
  
  def build_GPT_DB_from_directory(self, dir_name):
    for filename in os.listdir(dir_name):
      if filename.endswith(".txt"):
        print("trainning from file: ", filename)
        self.build_gpt_DB_from_file(dir_name + "/" + filename)
      else:
        print("passing: ", filename)
  def store_GPT_DB_to_json_file(self):
    with open(self.GPT_DB_FILE, "w") as file:
      json.dump(self.db, file)
    
  def load_GPT_DB_from_json_file(self):
    if not os.path.exists(self.GPT_DB_FILE):
      return False
    
    with open(self.GPT_DB_FILE, "r") as file:
      self.db = json.load(file)
      return True
 
gpt = GPT()
if gpt.load_GPT_DB_from_json_file() == False:
  gpt.build_GPT_DB_from_directory("data")
  gpt.store_GPT_DB_to_json_file()
  
gpt.infer("love", 10)
