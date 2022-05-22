import json 
import glob
from tqdm import tqdm

fout = open("/research/d4/gds/mindahu21/ssptGen/000/sspt_qg_combined_dataset.jsonl", "w")
for i in tqdm(range(100)):
    if i < 10:
        i_str = "00" + str(i)
    else:
        i_str = "0" + str(i)

    sspt_file = "/research/d4/gds/mindahu21/ssptGen/000/sspt_{}.jsonl".format(i_str)
    qg_file = "/research/d4/gds/mindahu21/ssptGen/000/qg_ans_aware_sspt_{}.jsonl".format(i_str)
    sspt_fin = open(sspt_file, 'r')
    qg_fin = open(qg_file, 'r')
    
    for sspt_line, qg_line in zip(sspt_fin, qg_fin):
        sspt_js = json.loads(sspt_line)
        qg_js = json.loads(qg_line)
        assert sspt_js['answers'][0] == qg_js['answer']
        print(json.dumps({"sspt_line": sspt_line, "moco_line": qg_line}), file = fout)
    
    sspt_fin.close()
    qg_fin.close()

fout.close()