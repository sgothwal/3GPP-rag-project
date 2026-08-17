from unstructured.partition.docx import partition_docx
from unstructured.chunking.title import chunk_by_title
from pathlib import Path
import json

all_chunks=[]
base_path = Path(__file__).parent.parent/"Data"


def docs_loder():
   
    print("---------------------- pipeline starting ----------------------\n\n")
    print("!!!STEPS!!!\n\n")
    print(" I-loading files from directory")
    data_folder = base_path
    files=list(data_folder.glob("*.docx"))
    print("II-!!✈️ sending loaded file for element creation!!\n\n")
    ele_loader(files)

def ele_loader(files)->tuple[list,str]:

    for i,file in enumerate(files):
        print(f"III-!!creating elements from file =>{file.name}- {i+1}/{len(files)}files!!\n\n")
       
        elements = partition_docx(
        filename=str(file), 
        infer_table_structure=True                
         )   
        
        print(f"IV-!!sending elements for chunks formation!!")
        chunker(elements,file.name)
def chunker(elements,file_name)->tuple[list,str]:
   """forms chunks"""
   print(f"V-!!starting chunking for file {file_name}!!\n\n")
   chunks = chunk_by_title(elements,
                        include_orig_elements=True,
                        max_characters=1200,
                        combine_text_under_n_chars=500,
                        new_after_n_chars=1000,
                        overlap=150)
   loader(chunks,file_name)
   print(elements[0],file_name)

def loader(chunks,file_name)->list:
    """Extracts data from chunks"""
    print(f"VI-!!starting extraction of data from chunks of file {file_name}!!\n\n")
    
    c_len=len(chunks)
    for i,chunk in enumerate(chunks):
        f_chunk=extractor(chunk,i+1,c_len)
        all_chunks.append(f_chunk)
        print(f"chunk {i+1}/{c_len} sent 🛩️ to final list \n\n" )    

def extractor(chunk,i,c_length)->dict:
    text=chunk.text
    chunk_id=chunk.id
    filename=chunk.metadata.filename
    parent_ids=set()
    table=[]
   
    print(f"processing chunk ⏳:{i}/{c_length}\n\n")
    for ele in chunk.metadata.orig_elements:
        parent_ids.add(ele.metadata.parent_id)
        ele_cat=ele.category
        
        try:
            if ele_cat=='Table':
                print(f"!!!found Table to Process in chunk {i}!!!\n\n")
                table_data=getattr(ele.metadata,'text_as_html','text')
                table.append(table_data)

            
        
        except Exception as e: 
            print("error",e)
     
    print(f"filename is {filename} \n chunk id is{chunk_id}_txt \n parent_id is {parent_ids}")
    return({"text":text,
                "table":table,
                "metadata":{"filename":filename,"parent_ids":list(parent_ids),"chunk_id":chunk_id},
            })   
def export(filename="../final_data.json",data=all_chunks)->json:
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print("!!!!! VII file is ready in your root folder 📁 !!!! \n\n\n")
            
if __name__== "__main__":
    docs_loder()
    export() 

