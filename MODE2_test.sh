# Build folder for saving searching results
folder_name=MODE3
folder=${folder_name}
mkdir -p ./log/${folder}

job_name=periodic_helmholtz
echo ${job_name}
python test_pde.py  ${job_name} ${folder}

job_name=jump_green
echo ${job_name}
python test_pde.py  ${job_name} ${folder}
