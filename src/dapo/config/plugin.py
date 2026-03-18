try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    import regex as re
    import numpy as np
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def format_score(model_output: str) -> float:
    pattern = r'^<think>\n\n<pre>(.*?)</pre>(.*?)</think>(.+)$'
    return 1.0 if re.match(pattern, model_output, re.DOTALL | re.MULTILINE) else 0.0


def compute_score_all(data_source, solution_str, ground_truth, extra_info=None) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    # Wrap the ground truth in \boxed{} format for verification
    if ground_truth.find('boxed') < 0:
        sol = "\\boxed{" + ground_truth + "}"
    else:
        sol = ground_truth

    try:
        pre_reward, _ = verify_func([sol], [solution_str.split('</pre>')[0]])
        post_reward, _ = verify_func([sol], [solution_str.split('</think>')[-1]])

        # format reward
        f_reward = format_score(solution_str)

        # accuracy reward
        a_reward = post_reward * (0.6 + 0.4 * pre_reward)
        
        # length reward
        cur_len = len(solution_str.split('</pre>')[-1])
        clipped_length = np.clip(cur_len, 1, 4000)
        normalized_length = (clipped_length - 1) / (4000 - 1)

        if pre_reward == 1.0:
            len_reward = 1.0 - (normalized_length ** 4.0) 
        elif post_reward == 0.0:
            len_reward = - (1.0 - (normalized_length ** 4.0))
        else:
            len_reward = 0.0

        reward = f_reward * a_reward + 0.01 * len_reward
    except Exception:
        reward = 0.0
    except TimeoutException:
        reward = 0.0

    return reward


def format_score_ori(model_output: str) -> float:
    pattern = r'^<think>(.*?)</think>(.+)$'
    return 1.0 if re.match(pattern, model_output, re.DOTALL | re.MULTILINE) else 0.0


def compute_score_ori(data_source, solution_str, ground_truth, extra_info=None) -> float:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    # Wrap the ground truth in \boxed{} format for verification
    if ground_truth.find('boxed') < 0:
        sol = "\\boxed{" + ground_truth + "}"
    else:
        sol = ground_truth

    try:
        post_reward, _ = verify_func([sol], [solution_str])

        f_score = format_score_ori(solution_str)

        reward = f_score * post_reward

        # reward = 0.99 * reward + 0.01 * f_score
    except Exception:
        reward = 0.0
    except TimeoutException:
        reward = 0.0

    return reward