#!/usr/bin/env python
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import os

# possible input args
folder = '<Insert folder path here>'
#out_folder = os.path.join(folder, "FastQC")
#if not os.path.exists(out_folder):
#	os.makedirs(out_folder)

#Possible_Amplicons
#CTTCCAACAACCGGAAGTGANNNNNNNNNNNNNNtggttacaaataaagc
#XXCTTCCAACAACCGGAAGTGANNNNNNNNNNNNNNtggttacaaataaa
#XXXXCTTCCAACAACCGGAAGTGANNNNNNNNNNNNNNtggttacaaata
#XXXXXXCTTCCAACAACCGGAAGTGANNNNNNNNNNNNNNtggttacaaa

R1_primer = "CTTCCAACAACCGGAAGTGA" #real
R1_downstream = "TGGTTACAAATAAAG"
os.chdir(folder)
data_file_names = os.listdir()

#Trim reads
for i in data_file_names:
    if i[-3:] == ".gz":
        c_file = i[:-9]
        R1 = c_file + "_trimmed.fastq"
        execute = "cutadapt -m 14 -M 14 -e 0.2 -q 20 -a " + R1_primer + "..." + R1_downstream + " " + i + " > " + R1
        os.system(execute)
    elif i[-5:] == "fastq":
        c_file = i[:-6]
        R1 = c_file + "_trimmed.fastq"
        execute = "cutadapt -m 14 -M 14 -e 0.2 -q 20 -a " + R1_primer + "..." + R1_downstream + " " + i + " > " + R1
        os.system(execute)

data_file_names = os.listdir()
files = []
for i in data_file_names:
    if i[-14:] == "_trimmed.fastq":
        fx = os.path.join(folder, i)
        files.append(fx)

file_list = []
barcode_list = []
num_reads_list = []
qual_list = []
notes_list = []
for i in files:
    fname = i
    num_reads = 0
    G_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    A_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    T_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    C_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    Seq_list = []
    Quality = []
    for record in SeqIO.parse(fname, "fastq"):
    	qual = np.array(record.letter_annotations["phred_quality"])
    	avg_qual = np.average(qual)
    	Quality.append(avg_qual)
    	num_reads += 1
    	xx = str(record.seq)
    	Seq_list.append(xx)
    	for j in range(14):
    		base = xx[j]
    		if base == "G":
    			G_list[j]+=1
    		elif base == "A":
    			A_list[j]+=1
    		elif base == "T":
    			T_list[j]+=1
    		elif base == "C":
    			C_list[j]+=1
    G_percent = [x/num_reads*100 for x in G_list]
    A_percent = [x/num_reads*100 for x in A_list]
    T_percent = [x/num_reads*100 for x in T_list]
    C_percent = [x/num_reads*100 for x in C_list]

    x=np.array(range(14))
    y=np.array(G_percent)
    fig, ax = plt.subplots()
    plt.xlabel('position')
    plt.ylabel('% base')
    plt.plot (x,y, color = 'black', label = "G")
    y=np.array(A_percent)
    plt.plot (x,y, color = 'green', label = "A")
    y=np.array(T_percent)
    plt.plot (x,y, color = 'red', label = "T")
    y=np.array(C_percent)
    plt.plot (x,y, color = 'blue', label = "C")
    #plt.legend()
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=4)
    fig.set_facecolor('white')
    fig_name = i + ".png"
    plt.savefig(fig_name)
    plt.clf()
    
    #Calling sequence
    Sequence_temp = []
    for k in range(14):
    	if G_percent[k]>80:
    		Sequence_temp.append('G')
    	elif A_percent[k]>80:
    		Sequence_temp.append('A')
    	elif T_percent[k]>80:
    		Sequence_temp.append('T')
    	elif C_percent[k]>80:
    		Sequence_temp.append('C')
    	else:
    		Sequence_temp.append("N")
    		#break
    	if num_reads < 500: #Require at least 500 reads
    		Barcode_temp = "Insufficient Data"
    		Notes_temp = 'Insufficient Data'	    	
    	elif "N" in Sequence_temp:
    		Barcode_temp = ''.join(str(e) for e in Sequence_temp)
    		Notes_temp = 'Mixed Sequence'
    	else:
    		Barcode_temp = ''.join(str(e) for e in Sequence_temp)
    		Notes_temp = ""
    avg_Quality = np.average(Quality)
    qual_list.append(avg_Quality)
    file_list.append(i)
    barcode_list.append(Barcode_temp)
    num_reads_list.append(str(num_reads))
    notes_list.append(Notes_temp)

#Writing out sequence info
#Make summary file and add headers
save_name = "UMGC_IL_073_Sequence_summary_REV2.txt"
save_file = open(save_name, "w")
header = ("Filename",'\t',"Total Reads",'\t',"Mean per sequence Q-score",'\t',"Sequence",'\t',"Notes",'\t',"SampleName",'\t',"InjectionPlate",'\t',"EmbryoID",'\t',"ProgenyID",'\n')
save_file.write(''.join(map(str, header)))
newtab = '\t'
newline = '\n'

embryo_ID_list = []
for i, item in enumerate(file_list):
	save_file.write(str(item))
	save_file.write(newtab)
	save_file.write(str(num_reads_list[i]))
	save_file.write(newtab)
	save_file.write(str(qual_list[i]))
	save_file.write(newtab)
	save_file.write(str(barcode_list[i]))
	save_file.write(newtab)	
	save_file.write(str(notes_list[i]))
	sample_name = os.path.splitext(os.path.basename(item))[0].split("_S")[0]
	injection_plate = sample_name.split("_")[0]
	if sample_name.split("_")[0][0:3] == "BR1":
		embryo_ID = "BR_1"
	elif sample_name.split("_")[0][0:3] == "BR2":
		embryo_ID = "BR_2"
	elif sample_name.split("_")[0][0:3] == "BR3":
		embryo_ID = "BR_3"		
	else:
		embryo_ID = sample_name.split("_")[0] + "_" + sample_name.split("_")[1]
	Progeny_ID = sample_name.split("_")[2]
	embryo_ID_list.append(embryo_ID)	
	save_file.write(newtab)	
	save_file.write(str(sample_name))
	save_file.write(newtab)	
	save_file.write(str(injection_plate))
	save_file.write(newtab)	
	save_file.write(str(embryo_ID))
	save_file.write(newtab)	
	save_file.write(str(Progeny_ID))
	save_file.write(newline)
	
save_file.close()


unique_embryo_IDs = set(embryo_ID_list)

barcodes_per_embryo = []
embryo = []
barcodes_by_embryo = []
for i in (unique_embryo_IDs):
	temp_barcode_list = []
	for j, item in enumerate(file_list):
		sample_name = os.path.splitext(os.path.basename(item))[0].split("_S")[0]
		injection_plate = sample_name.split("_")[0]
		if sample_name.split("_")[0][0:3] == "BR1":
			embryo_ID = "BR_1"
		elif sample_name.split("_")[0][0:3] == "BR2":
			embryo_ID = "BR_2"
		elif sample_name.split("_")[0][0:3] == "BR3":
			embryo_ID = "BR_3"		
		else:
			embryo_ID = sample_name.split("_")[0] + "_" + sample_name.split("_")[1]
		if i == embryo_ID:
			if notes_list[j] != 'Mixed Sequence' and notes_list[j] != 'Insufficient Data':
				temp_barcode_list.append(barcode_list[j])
	embryo.append(i)
	unique_barcodes = set(temp_barcode_list)
	barcodes_by_embryo.append(unique_barcodes)
	barcodes_per_embryo.append(len(unique_barcodes))

#Culling the stock collection and prioritizing injected flies for additional crosses
#Make summary file and add headers
save_name = "UMGC_IL_073_Stock collection.txt"
save_file = open(save_name, "w")
header = ("Filename",'\t',"Total Reads",'\t',"Mean per sequence Q-score",'\t',"Sequence",'\t',"Notes",'\t',"SampleName",'\t',"InjectionPlate",'\t',"EmbryoID",'\t',"ProgenyID",'\t',"Barcodes_per_embryo",'\t',"Keep?",'\n')
save_file.write(''.join(map(str, header)))
newtab = '\t'
newline = '\n'

embryo_ID_list = []
unique_barcodes_so_far = []
for i, item in enumerate(file_list):
	save_file.write(str(item))
	save_file.write(newtab)
	save_file.write(str(num_reads_list[i]))
	save_file.write(newtab)
	save_file.write(str(qual_list[i]))
	save_file.write(newtab)
	save_file.write(str(barcode_list[i]))
	save_file.write(newtab)	
	save_file.write(str(notes_list[i]))
	sample_name = os.path.splitext(os.path.basename(item))[0].split("_S")[0]
	injection_plate = sample_name.split("_")[0]
	if sample_name.split("_")[0][0:3] == "BR1":
		embryo_ID = "BR_1"
	elif sample_name.split("_")[0][0:3] == "BR2":
		embryo_ID = "BR_2"
	elif sample_name.split("_")[0][0:3] == "BR3":
		embryo_ID = "BR_3"		
	else:
		embryo_ID = sample_name.split("_")[0] + "_" + sample_name.split("_")[1]
	Progeny_ID = sample_name.split("_")[2]
	embryo_ID_list.append(embryo_ID)	
	save_file.write(newtab)	
	save_file.write(str(sample_name))
	save_file.write(newtab)	
	save_file.write(str(injection_plate))
	save_file.write(newtab)	
	save_file.write(str(embryo_ID))
	save_file.write(newtab)	
	save_file.write(str(Progeny_ID))
	save_file.write(newtab)	
	for j, item2 in enumerate(embryo):
		if item2 == embryo_ID:
			save_file.write(str(barcodes_per_embryo[j]))
			save_file.write(newtab)
	if str(barcode_list[i]) not in unique_barcodes_so_far and str(notes_list[i]) != "Mixed Sequence" and str(notes_list[i]) != "Insufficient Data":
		unique_barcodes_so_far.append(barcode_list[i])
		save_file.write("Yes")
		#save_file.write(newtab)
	else:
		save_file.write("No")
		#save_file.write(newtab)	
	save_file.write(newline)
		
save_file.close()


#Plotting a histogram
newvalues = [x for x in barcodes_per_embryo if x != 0]
a = np.array(newvalues)
 
# Creating histogram
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a, bins = [0,1,2,3,4,5,6,7,8,9,10])
 
# Show plot
plt.show()


