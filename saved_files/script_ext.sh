#!/bin/bash                                                                                        
#SBATCH --job-name Ext_2ndhalf                                                                    
#SBATCH --partition=longq                                                                       
#SBATCH --nodes=1 # number of nodes                                                                
#SBATCH --mem 70000M # memory pool for all cores   
##SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
echo "parallel extracting data second half"
python parallel_extract_keywords.py
