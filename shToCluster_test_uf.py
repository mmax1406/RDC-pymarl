import re

# Paste your full list of sh ./test_uf.sh lines here
raw_input = """
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++MPE+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------------------------------------------------TAG-------------------------------------------------------------
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_b.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-N" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_n.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-DA" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_da.log'

#####transformer#####
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms_h.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h.log'
# teacher-student
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="TAG-QMIX-TRANS-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_student.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="TAG-QMIX-TRANS-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-TRANS-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ms_h_tf.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-TRANS-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_trans_ss_h_tf.log'

#####gru#####
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" exp_name="TAG-QMIX-GRU-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" exp_name="TAG-QMIX-GRU-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" use_history=True exp_name="TAG-QMIX-GRU-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms_h.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True exp_name="TAG-QMIX-GRU-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h.log'
# teacher-student
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" exp_name="TAG-QMIX-GRU-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" exp_name="TAG-QMIX-GRU-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_student.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True exp_name="TAG-QMIX-GRU-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True exp_name="TAG-QMIX-GRU-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" use_history=True exp_name="TAG-QMIX-GRU-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="multi_step" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-GRU-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ms_h_tf.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="single_step" use_history=True teacher_forcing_start_value=1.0 exp_name="TAG-QMIX-GRU-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_gru_ss_h_tf.log'


# -------------------------------------------------------------SPREAD-------------------------------------------------------------
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="none" exp_name="SPREAD-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_b.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="none" exp_name="SPREAD-QMIX-N" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_n.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="none" exp_name="SPREAD-QMIX-DA" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_da.log'

#####transformer#####
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms_h.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h.log'
# teacher-student
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="SPREAD-QMIX-TRANS-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_student.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="SPREAD-QMIX-TRANS-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-TRANS-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ms_h_tf.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-TRANS-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_trans_ss_h_tf.log'

#####gru#####
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" exp_name="SPREAD-QMIX-GRU-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" exp_name="SPREAD-QMIX-GRU-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" use_history=True exp_name="SPREAD-QMIX-GRU-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms_h.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True exp_name="SPREAD-QMIX-GRU-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h.log'
# teacher-student
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" exp_name="SPREAD-QMIX-GRU-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" exp_name="SPREAD-QMIX-GRU-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_student.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True exp_name="SPREAD-QMIX-GRU-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True exp_name="SPREAD-QMIX-GRU-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" use_history=True exp_name="SPREAD-QMIX-GRU-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="multi_step" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-GRU-MS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ms_h_tf.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-spread-v3" predictor_mode="single_step" use_history=True teacher_forcing_start_value=1.0 exp_name="SPREAD-QMIX-GRU-SS-H-TF" test_nepisode=1280 evaluate=True checkpoint_path=""' 'spread_test_qmix_gru_ss_h_tf.log'


-------------------------------------------------------------REFERENCE-------------------------------------------------------------
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="none" exp_name="REFERENCE-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_b.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="none" exp_name="REFERENCE-QMIX-N" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_n.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="none" exp_name="REFERENCE-QMIX-DA" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_da.log'

#####transformer#####
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" exp_name="REFERENCE-QMIX-TRANS-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ms.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="REFERENCE-QMIX-TRANS-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ss.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="REFERENCE-QMIX-TRANS-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ms_h.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="REFERENCE-QMIX-TRANS-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ss_h.log'
# teacher-student
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="REFERENCE-QMIX-TRANS-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ss_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" exp_name="REFERENCE-QMIX-TRANS-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ss_student.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="REFERENCE-QMIX-TRANS-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ss_h_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" transformer_structure="encoder-decoder" use_history=True exp_name="REFERENCE-QMIX-TRANS-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ss_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_tf4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="multi_step" transformer_structure="encoder-decoder" use_history=True exp_name="REFERENCE-QMIX-TRANS-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_trans_ms_h_student.log'

#####gru#####
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="multi_step" exp_name="REFERENCE-QMIX-GRU-MS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ms.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" exp_name="REFERENCE-QMIX-GRU-SS" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ss.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="multi_step" use_history=True exp_name="REFERENCE-QMIX-GRU-MS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ms_h.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" use_history=True exp_name="REFERENCE-QMIX-GRU-SS-H" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ss_h.log'
# teacher-student
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" exp_name="REFERENCE-QMIX-GRU-SS-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ss_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" exp_name="REFERENCE-QMIX-GRU-SS-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ss_student.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" use_history=True exp_name="REFERENCE-QMIX-GRU-SS-H-TEACHER" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ss_h_teacher.log'
sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="single_step" use_history=True exp_name="REFERENCE-QMIX-GRU-SS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ss_h_student.log'

sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="1" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-reference-v3" predictor_mode="multi_step" use_history=True exp_name="REFERENCE-QMIX-GRU-MS-H-STUDENT" test_nepisode=1280 evaluate=True checkpoint_path=""' 'reference_test_qmix_gru_ms_h_student.log'
"""

def clean_base_command(cmd):
    """
    Strips out nohup, CUDA environment variables, and empty checkpoint paths
    to make the command safe for SLURM.
    """
    # Remove CUDA_VISIBLE_DEVICES="X"
    cmd = re.sub(r'CUDA_VISIBLE_DEVICES="[^"]*"\s+', '', cmd)
    # Remove nohup
    cmd = cmd.replace('nohup ', '')
    # Remove the dummy checkpoint_path=""
    cmd = cmd.replace('checkpoint_path=""', '')
    # Remove existing evaluate=True (we add it back once for consistency)
    cmd = cmd.replace('evaluate=True', '')
    
    return cmd.strip().rstrip('&').strip()

def generate_uf_slurm_list(input_text, output_file="experiments_uf.txt"):
    lines = input_text.strip().split('\n')
    total_commands = 0
    
    with open(output_file, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Extract command and log file
            match = re.search(r"sh \./test_uf\.sh\s+'([^']+)'\s+'([^']+)'", line)
            if not match:
                continue
                
            base_cmd = match.group(1)
            log_file = match.group(2)

            # Extract exp_name for the dynamic checkpoint lookup
            exp_match = re.search(r'exp_name="([^"]+)"', base_cmd)
            exp_name = exp_match.group(1) if exp_match else "unknown"

            # THE PATH FIX: Search results/models directly (no dot) to get clean relative paths
            find_ckpt = f'$(find results/models -type d -name "*{exp_name}*" 2>/dev/null | sort -r | head -n 1)'

            clean_cmd = clean_base_command(base_cmd)

            # The loop logic: i=0 (In-dist) and i=1 (Out-of-dist)
            for i in [0, 1]:
                if i == 0:
                    delay_val, delay_scope = 6, 3
                else:
                    delay_val, delay_scope = 9, 3

                # n_expand_action logic for unfixed delay
                if any(x in log_file for x in ["ss_h", "ms_h"]):
                    n_expand = 9
                else:
                    n_expand = delay_val + delay_scope
                
                # Assemble the single line
                final_line = (
                    f"{clean_cmd} "
                    f"checkpoint_path=\"{find_ckpt}\" "
                    f"evaluate=True delay_type=\"uf\" "
                    f"delay_value={delay_val} delay_scope={delay_scope} "
                    f"n_expand_action={n_expand} "
                    f">> {log_file} 2>&1"
                )
                
                f.write(final_line + "\n")
                total_commands += 1

    print(f"--- SUCCESS ---")
    print(f"Generated {output_file} with {total_commands} lines.")
    print(f"Update your SLURM header: #SBATCH --array=1-{total_commands}%5")

if __name__ == "__main__":
    generate_uf_slurm_list(raw_input)