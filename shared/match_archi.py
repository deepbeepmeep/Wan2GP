import re

def match_nvidia_architecture(conditions_dict, architecture):
    """
    Match Nvidia architecture against condition dictionary.
    
    Args:
        conditions_dict: dict with condition strings as keys, parameters as values
        architecture: int representing architecture (e.g., 89 for Ada Lovelace)
    
    Returns:
        list of matched parameters
    
    Condition syntax:
        - Operators: '<', '>', '<=', '>=', '=' (or no operator for equality)
        - OR: '+' between conditions (e.g., '<=50+>89')
        - AND: '&' between conditions (e.g., '>=70&<90')
        - Examples:
          * '<89': architectures below Ada (89)
          * '>=75': architectures 75 and above
          * '89': exactly Ada architecture
          * '<=50+>89': Maxwell (50) and below OR above Ada
          * '>=70&<90': Ampere range (70-89)
    """
    
    def eval_condition(cond, arch):
        """Evaluate single condition against architecture"""
        cond = cond.strip()
        if not cond:
            return False
            
        # Parse operator and value using regex
        match = re.match(r'(>=|<=|>|<|=?)(\d+)', cond)
        if not match:
            return False
            
        op, val = match.groups()
        val = int(val)
        
        # Handle operators
        if op in ('', '='):
            return arch == val
        elif op == '>=':
            return arch >= val
        elif op == '<=':
            return arch <= val
        elif op == '>':
            return arch > val
        elif op == '<':
            return arch < val
        return False
    
    def matches_condition(condition_str, arch):
        """Check if architecture matches full condition string"""
        # Split by '+' for OR conditions, then by '&' for AND conditions
        return any(
            all(eval_condition(and_cond, arch) for and_cond in or_cond.split('&'))
            for or_cond in condition_str.split('+')
            if or_cond.strip()
        )
    
    # Return all parameters where conditions match
    return [params for condition, params in conditions_dict.items() 
            if matches_condition(condition, architecture)]