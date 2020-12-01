# Susanna Ardigò
## Knowledge Analysis and Management
### Project 2: Multi-source code search
#### Run the project with scripts
To run the project use the command:
    `sh run.sh path_to_tensorflow_directory "query"`
   
    
 #### Run single files
Extract data:
* `python3 src/extract_data.py path_to_tensorflow_directory`

Search data:
* `python3 src/search_data.py "query"`

Evaluate:
* `python3 src/prec_recall.py path_to_csv_directory path_to_ground_truth`

