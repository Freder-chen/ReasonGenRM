# Inspired by https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/pair-pm/process_pair_data.py
def is_valid_item(item):
    """
    Check if the dataset item is valid based on the structure of the chosen and rejected messages.
    Return True if valid, False otherwise.
    """
    chosen, rejected = item.get('chosen'), item.get('rejected')

    # Validate the length of chosen and rejected lists
    if not chosen or not rejected:
        return False
    if len(chosen) != len(rejected) or len(chosen) % 2 != 0:
        return False

    n_rounds = len(chosen)
    roles = ['user', 'assistant']

    for i in range(n_rounds):
        # Check roles are iteratively 'user' and 'assistant'
        expected_role = roles[i % 2]
        if chosen[i]['role'] != expected_role or rejected[i]['role'] != expected_role:
            return False

        # Check content is non-empty
        if not chosen[i]['content'] or not rejected[i]['content']:
            return False

        # Check context consistency for all but the last round
        if i < n_rounds - 1 and chosen[i]['content'] != rejected[i]['content']:
            return False

        # Check the last round has different content
        if i == n_rounds - 1 and chosen[i]['content'] == rejected[i]['content']:
            return False

    return True
