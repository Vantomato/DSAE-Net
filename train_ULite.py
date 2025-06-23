# Define the commands to run the scripts with the desired arguments
import subprocess
model_name = 'ULite'
datasets = ['DRIVE', 'STARE', 'CHASEDB1', 'HRF']
cycle_lens = '20/10'
def run_pipeline(model_name, cycle_lens, dataset):
    commands = [
        # [
        #     'python', 'train_cyclical.py',
        #     '--csv_train', f'data/{dataset}/train.csv',
        #     '--model_name', f'{model_name}',
        #     '--cycle_lens', f'{cycle_lens}',
        #     '--save_path', f'{model_name}_{dataset}'
        # ],
        [
            'python', 'generate_results.py',
            '--config_file', f'experiments/{model_name}_{dataset}/config.cfg',
            '--dataset', f'{dataset}'
        ],
        [
            'python', 'analyze_results.py',
            '--path_train_preds', f'results/{dataset}/experiments/{model_name}_{dataset}',
            '--path_test_preds', f'results/{dataset}/experiments/{model_name}_{dataset}',
            '--train_dataset', f'{dataset}',
            '--test_dataset', f'{dataset}'
        ]
    ]

    # Run the commands in sequence
    for command in commands:
        subprocess.run(command)

for dataset in datasets:
    run_pipeline(model_name, cycle_lens, dataset)