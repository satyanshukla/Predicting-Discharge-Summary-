#!/bin/bash                                                                                        
#SBATCH --job-name Ext_2ndhalf                                                                    
#SBATCH --partition=longq                                                                       
#SBATCH --nodes=1 # number of nodes                                                                
#SBATCH --mem 40000M # memory pool for all cores   
#SBATCH --ntasks-per-node=2
echo "predict notes"
python predict_notes.py
