import re

# Add your raw data here
raw_input = """
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++MPE+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------------------------------------------TAG-------------------------------------------------------------
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_b.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-N" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_n.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-DA" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_da.log'

#####transformer#####
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms_h.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h.log'
# teacher-student
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_student.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-TRANS-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms_h_tf.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-TRANS-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h_tf.log'

#####gru#####
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" exp_name="TAG-QMIX-GRU-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" exp_name="TAG-QMIX-GRU-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" use_history=True exp_name="TAG-QMIX-GRU-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms_h.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True exp_name="TAG-QMIX-GRU-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h.log'
# teacher-student
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" exp_name="TAG-QMIX-GRU-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" exp_name="TAG-QMIX-GRU-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_student.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True exp_name="TAG-QMIX-GRU-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True exp_name="TAG-QMIX-GRU-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" use_history=True exp_name="TAG-QMIX-GRU-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-GRU-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms_h_tf.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-GRU-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h_tf.log'


# -------------------------------------------------------------SPREAD-------------------------------------------------------------
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="none" exp_name="SPREAD-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_b.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="none" exp_name="SPREAD-QMIX-N" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_n.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="none" exp_name="SPREAD-QMIX-DA" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_da.log'

#####transformer#####
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms_h.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h.log'
# teacher-student
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_student.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-TRANS-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms_h_tf.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-TRANS-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h_tf.log'

#####gru#####
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" exp_name="SPREAD-QMIX-GRU-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" exp_name="SPREAD-QMIX-GRU-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" use_history=True exp_name="SPREAD-QMIX-GRU-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms_h.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True exp_name="SPREAD-QMIX-GRU-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h.log'
# teacher-student
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" exp_name="SPREAD-QMIX-GRU-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" exp_name="SPREAD-QMIX-GRU-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_student.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True exp_name="SPREAD-QMIX-GRU-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h_teacher.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True exp_name="SPREAD-QMIX-GRU-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" use_history=True exp_name="SPREAD-QMIX-GRU-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms_h_student.log'

sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-GRU-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms_h_tf.log'
sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="1" nohup python3 -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-GRU-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h_tf.log'

"""

def clean_base_command(cmd):
    """
    Strips out nohup, CUDA environment variables, and empty checkpoint paths
    to make the command safe for SLURM.
    """
    # Remove CUDA_VISIBLE_DEVICES (e.g., CUDA_VISIBLE_DEVICES="1")
    cmd = re.sub(r'CUDA_VISIBLE_DEVICES="[^"]*"\s+', '', cmd)
    # Remove nohup
    cmd = cmd.replace('nohup ', '')
    # Remove the dummy checkpoint_path=""
    cmd = cmd.replace('checkpoint_path=""', '')
    # Remove extra evaluate=True to keep the line clean (we add it back once)
    cmd = cmd.replace('evaluate=True', '')
    
    return cmd.strip()

def generate_slurm_list(input_text, output_file="experiments.txt"):
    lines = input_text.strip().split('\n')
    total_commands = 0
    
    with open(output_file, 'w') as f:
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Extract the command and the log file
            match = re.search(r"sh \./test_f\.sh\s+'([^']+)'\s+'([^']+)'", line)
            if not match:
                continue
                
            base_cmd = match.group(1)
            log_file = match.group(2)

            # Extract exp_name for checkpoint finding
            exp_match = re.search(r'exp_name="([^"]+)"', base_cmd)
            exp_name = exp_match.group(1) if exp_match else "unknown"

            # Checkpoint Logic: Find the latest directory matching the exp_name
            # We search results/models directly to avoid the leading './' issue
            find_ckpt = f'$(find results/models -type d -name "*{exp_name}*" 2>/dev/null | sort -r | head -n 1)'

            clean_cmd = clean_base_command(base_cmd)

            # Generate 13 lines for fixed delays (0 to 12)
            for delay in range(13):
                # Expansion logic: ss_h/ms_h get 9 if delay > 0, else 0. 
                # Others follow the delay value.
                if any(x in log_file for x in ["ss_h", "ms_h"]):
                    n_expand = 0 if delay == 0 else 9
                else:
                    n_expand = delay
                
                # Assemble the final command line
                final_line = (
                    f"{clean_cmd} "
                    f"checkpoint_path=\"{find_ckpt}\" "
                    f"evaluate=True delay_type=\"f\" "
                    f"delay_value={delay} delay_scope=0 "
                    f"n_expand_action={n_expand} "
                    f">> {log_file} 2>&1"
                )
                
                f.write(final_line + "\n")
                total_commands += 1

    print(f"--- SUCCESS ---")
    print(f"Created {output_file} with {total_commands} total command lines.")
    print(f"Update your SLURM header to: #SBATCH --array=1-{total_commands}%5")

if __name__ == "__main__":
    generate_slurm_list(raw_input)