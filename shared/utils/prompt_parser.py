import re

def process_template(input_text, keep_comments = False):
    """
    Process a text template with macro instructions and variable substitution.
    Supports multiple values for variables to generate multiple output versions.
    Each section between macro lines is treated as a separate template.
    
    Args:
        input_text (str): The input template text
        
    Returns:
        tuple: (output_text, error_message)
            - output_text: Processed output with variables substituted, or empty string if error
            - error_message: Error description and problematic line, or empty string if no error
    """
    lines = input_text.strip().split('\n')
    current_variables = {}
    current_template_lines = []
    all_output_lines = []
    error_message = ""
    
    # Process the input line by line
    line_number = 0
    while line_number < len(lines):
        orig_line = lines[line_number]
        line = orig_line.strip()
        line_number += 1
        
        # Skip empty lines or comments
        if not line:
            continue

        if line.startswith('#') and not keep_comments:
            continue

        # Handle macro instructions
        if line.startswith('!'):
            # Process any accumulated template lines before starting a new macro
            if current_template_lines:
                # Process the current template with current variables
                template_output, err = process_current_template(current_template_lines, current_variables)
                if err:
                    return "", err
                all_output_lines.extend(template_output)
                current_template_lines = []  # Reset template lines
            
            # Reset variables for the new macro
            current_variables = {}
            
            # Parse the macro line
            macro_line = line[1:].strip()
            
            # Check for unmatched braces in the whole line
            open_braces = macro_line.count('{')
            close_braces = macro_line.count('}')
            if open_braces != close_braces:
                error_message = f"Unmatched braces: {open_braces} opening '{{' and {close_braces} closing '}}' braces\nLine: '{orig_line}'"
                return "", error_message
            
            # Check for unclosed quotes
            if macro_line.count('"') % 2 != 0:
                error_message = f"Unclosed double quotes\nLine: '{orig_line}'"
                return "", error_message
            
            # Split by optional colon separator
            var_sections = re.split(r'\s*:\s*', macro_line)
            
            for section in var_sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Extract variable name
                var_match = re.search(r'\{([^}]+)\}', section)
                if not var_match:
                    if '{' in section or '}' in section:
                        error_message = f"Malformed variable declaration\nLine: '{orig_line}'"
                        return "", error_message
                    continue
                    
                var_name = var_match.group(1).strip()
                if not var_name:
                    error_message = f"Empty variable name\nLine: '{orig_line}'"
                    return "", error_message
                
                # Check variable value format
                value_part = section[section.find('}')+1:].strip()
                if not value_part.startswith('='):
                    error_message = f"Missing '=' after variable '{{{var_name}}}'\nLine: '{orig_line}'"
                    return "", error_message
                
                # Extract all quoted values
                var_values = re.findall(r'"([^"]*)"', value_part)
                
                # Check if there are values specified
                if not var_values:
                    error_message = f"No quoted values found for variable '{{{var_name}}}'\nLine: '{orig_line}'"
                    return "", error_message
                
                # Check for missing commas between values
                # Look for patterns like "value""value" (missing comma)
                if re.search(r'"[^,]*"[^,]*"', value_part):
                    error_message = f"Missing comma between values for variable '{{{var_name}}}'\nLine: '{orig_line}'"
                    return "", error_message
                
                # Store the variable values
                current_variables[var_name] = var_values
        
        # Handle template lines
        else:
            if not line.startswith('#'):
                # Check for unknown variables in template line
                var_references = re.findall(r'\{([^}]+)\}', line)
                for var_ref in var_references:
                    if var_ref not in current_variables:
                        error_message = f"Unknown variable '{{{var_ref}}}' in template\nLine: '{orig_line}'"
                        return "", error_message
                
            # Add to current template lines
            current_template_lines.append(line)
    
    # Process any remaining template lines
    if current_template_lines:
        template_output, err = process_current_template(current_template_lines, current_variables)
        if err:
            return "", err
        all_output_lines.extend(template_output)
    
    return '\n'.join(all_output_lines), ""

def process_current_template(template_lines, variables):
    """
    Process a set of template lines with the current variables.
    
    Args:
        template_lines (list): List of template lines to process
        variables (dict): Dictionary of variable names to lists of values
        
    Returns:
        tuple: (output_lines, error_message)
    """
    if not variables or not template_lines:
        return template_lines, ""
    
    output_lines = []
    
    # Find the maximum number of values for any variable
    max_values = max(len(values) for values in variables.values())
    
    # Generate each combination
    for i in range(max_values):
        for template in template_lines:
            output_line = template
            for var_name, var_values in variables.items():
                # Use modulo to cycle through values if needed
                value_index = i % len(var_values)
                var_value = var_values[value_index]
                output_line = output_line.replace(f"{{{var_name}}}", var_value)
            output_lines.append(output_line)
    
    return output_lines, ""


def extract_variable_names(macro_line):
    """
    Extract all variable names from a macro line.
    
    Args:
        macro_line (str): A macro line (with or without the leading '!')
        
    Returns:
        tuple: (variable_names, error_message)
            - variable_names: List of variable names found in the macro
            - error_message: Error description if any, empty string if no error
    """
    # Remove leading '!' if present
    if macro_line.startswith('!'):
        macro_line = macro_line[1:].strip()
    
    variable_names = []
    
    # Check for unmatched braces
    open_braces = macro_line.count('{')
    close_braces = macro_line.count('}')
    if open_braces != close_braces:
        return [], f"Unmatched braces: {open_braces} opening '{{' and {close_braces} closing '}}' braces"
    
    # Split by optional colon separator
    var_sections = re.split(r'\s*:\s*', macro_line)
    
    for section in var_sections:
        section = section.strip()
        if not section:
            continue
            
        # Extract variable name
        var_matches = re.findall(r'\{([^}]+)\}', section)
        for var_name in var_matches:
            new_var = var_name.strip()
            if not new_var in variable_names: 
                variable_names.append(new_var)

    return variable_names, ""

def extract_variable_values(macro_line):
    """
    Extract all variable names and their values from a macro line.
    
    Args:
        macro_line (str): A macro line (with or without the leading '!')
        
    Returns:
        tuple: (variables_dict, error_message)
            - variables_dict: Dictionary mapping variable names to their values
            - error_message: Error description if any, empty string if no error
    """
    # Remove leading '!' if present
    if macro_line.startswith('!'):
        macro_line = macro_line[1:].strip()
    
    variables = {}
    
    # Check for unmatched braces
    open_braces = macro_line.count('{')
    close_braces = macro_line.count('}')
    if open_braces != close_braces:
        return {}, f"Unmatched braces: {open_braces} opening '{{' and {close_braces} closing '}}' braces"
    
    # Check for unclosed quotes
    if macro_line.count('"') % 2 != 0:
        return {}, "Unclosed double quotes"
    
    # Split by optional colon separator
    var_sections = re.split(r'\s*:\s*', macro_line)
    
    for section in var_sections:
        section = section.strip()
        if not section:
            continue
            
        # Extract variable name
        var_match = re.search(r'\{([^}]+)\}', section)
        if not var_match:
            if '{' in section or '}' in section:
                return {}, "Malformed variable declaration"
            continue
            
        var_name = var_match.group(1).strip()
        if not var_name:
            return {}, "Empty variable name"
        
        # Check variable value format
        value_part = section[section.find('}')+1:].strip()
        if not value_part.startswith('='):
            return {}, f"Missing '=' after variable '{{{var_name}}}'"
        
        # Extract all quoted values
        var_values = re.findall(r'"([^"]*)"', value_part)
        
        # Check if there are values specified
        if not var_values:
            return {}, f"No quoted values found for variable '{{{var_name}}}'"
        
        # Check for missing commas between values
        if re.search(r'"[^,]*"[^,]*"', value_part):
            return {}, f"Missing comma between values for variable '{{{var_name}}}'"
        
        variables[var_name] = var_values
    
    return variables, ""

def generate_macro_line(variables_dict):
    """
    Generate a macro line from a dictionary of variable names and their values.
    
    Args:
        variables_dict (dict): Dictionary mapping variable names to lists of values
        
    Returns:
        str: A formatted macro line (including the leading '!')
    """
    sections = []
    
    for var_name, values in variables_dict.items():
        # Format each value with quotes
        quoted_values = [f'"{value}"' for value in values]
        # Join values with commas
        values_str = ','.join(quoted_values)
        # Create the variable assignment
        section = f"{{{var_name}}}={values_str}"
        sections.append(section)
    
    # Join sections with a colon and space for readability
    return "! " + " : ".join(sections)