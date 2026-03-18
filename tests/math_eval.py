from evalscope import TaskConfig, run_task

prompt_template_no = "{query}"


def eval_math_500():
    # math500
    task_config = TaskConfig(
        api_url='http://127.0.0.1:8002/v1/chat/completions',
        model='Rethinker',
        eval_type='service',
        datasets=['math_500'],
        dataset_args={'math_500': {
            'few_shot_num': 0, 
            'subset_list': ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5'],
            'prompt_template': prompt_template_no,
        }},
        eval_batch_size=128,
        generation_config={
            'max_tokens': 7680,
            'temperature': 1.0,
            'n': 1,
            # 'chat_template_kwargs': {'enable_thinking': False}
        },
        work_dir='./output',
        ignore_errors=True,
        stream=True,
    )

    run_task(task_config)

def eval_aime_2025():
    # AIME-2025
    task_config = TaskConfig(
        api_url='http://127.0.0.1:8002/v1/chat/completions',
        model='Rethinker',
        eval_type='service', 
        datasets=['aime25'],
        dataset_args={
            'aime25': {
                    'few_shot_num': 0,
                    'subset_list': ['AIME2025-II', 'AIME2025-I'],
                    'prompt_template': prompt_template_no,
                    }
        },
        eval_batch_size=128,
        generation_config={
            'max_tokens': 7680,
            'temperature': 1.0,
            'n': 16,
            # 'chat_template_kwargs': {'enable_thinking': False}
        },
        work_dir='./output',
        ignore_errors=True,
        stream=True,
    )

    run_task(task_config)


def eval_gsm8k():
    # gsm8k
    task_config = TaskConfig(
        api_url='http://127.0.0.1:8002/v1/chat/completions',
        model='Rethinker',
        eval_type='service',
        datasets=['gsm8k'],
        dataset_args={'gsm8k': {
            'few_shot_num': 0, 
            'subset_list': ['main'],
            'prompt_template': prompt_template_no,
        }},
        eval_batch_size=128,
        generation_config={
            'max_tokens': 7680,
            'temperature': 1.0,
            'n': 1,
            # 'chat_template_kwargs': {'enable_thinking': False}
        },
        work_dir='./output',
        ignore_errors=True,
        stream=True,
    )

    run_task(task_config)



if __name__ == "__main__":
    eval_math_500()
