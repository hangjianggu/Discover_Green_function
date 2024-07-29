# Build folder for saving searching results
folder_name=MODE2
folder=${folder_name}
mkdir -p ./log/${folder}

job_name=negative_helmholtz_noise_5
echo ${job_name}
python test_pde.py  ${job_name} ${folder}

job_name=negative_helmholtz_noise_10
echo ${job_name}
python test_pde.py  ${job_name} ${folder}