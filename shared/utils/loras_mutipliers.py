def preparse_loras_multipliers(loras_multipliers):
    if isinstance(loras_multipliers, list):
        return [multi.strip(" \r\n") if isinstance(multi, str) else multi for multi in loras_multipliers]

    loras_multipliers = loras_multipliers.strip(" \r\n")
    loras_mult_choices_list = loras_multipliers.replace("\r", "").split("\n")
    loras_mult_choices_list = [multi.strip() for multi in loras_mult_choices_list if len(multi)>0 and not multi.startswith("#")]
    loras_multipliers = " ".join(loras_mult_choices_list)
    return loras_multipliers.split(" ")

def expand_slist(slists_dict, mult_no, num_inference_steps, model_switch_step, model_switch_step2 ):
    def expand_one(slist, num_inference_steps):
        if not isinstance(slist, list): slist = [slist]
        new_slist= []
        if num_inference_steps <=0:
            return new_slist
        inc =  len(slist) / num_inference_steps 
        pos = 0
        for i in range(num_inference_steps):
            new_slist.append(slist[ int(pos)])
            pos += inc
        return new_slist

    phase1 = slists_dict["phase1"][mult_no]
    phase2 = slists_dict["phase2"][mult_no]
    phase3 = slists_dict["phase3"][mult_no]
    shared = slists_dict["shared"][mult_no]
    if shared:
        if isinstance(phase1, float): return phase1
        return expand_one(phase1, num_inference_steps)    
    else:
        if isinstance(phase1, float) and isinstance(phase2, float) and isinstance(phase3, float) and phase1 == phase2 and phase2 == phase3: return phase1 
        return expand_one(phase1, model_switch_step) + expand_one(phase2, model_switch_step2 - model_switch_step) + expand_one(phase3, num_inference_steps - model_switch_step2)

def parse_loras_multipliers(loras_multipliers, nb_loras, num_inference_steps, merge_slist = None, nb_phases = 2, model_switch_step = None, model_switch_step2 = None):
    if model_switch_step is None:
        model_switch_step = num_inference_steps
    if model_switch_step2 is None:
        model_switch_step2 = num_inference_steps
    def is_float(element: any) -> bool:
        if element is None: 
            return False
        try:
            float(element)
            return True
        except ValueError:
            return False
    loras_list_mult_choices_nums = []
    slists_dict = { "model_switch_step": model_switch_step}
    slists_dict = { "model_switch_step2": model_switch_step2}
    slists_dict["phase1"] = phase1 = [1.] * nb_loras
    slists_dict["phase2"] = phase2 = [1.] * nb_loras
    slists_dict["phase3"] = phase3 = [1.] * nb_loras
    slists_dict["shared"] = shared = [False] * nb_loras

    if isinstance(loras_multipliers, list) or len(loras_multipliers) > 0:
        list_mult_choices_list = preparse_loras_multipliers(loras_multipliers)[:nb_loras]
        for i, mult in enumerate(list_mult_choices_list):
            current_phase = phase1
            if isinstance(mult, str):
                mult = mult.strip()
                phase_mult = mult.split(";")
                shared_phases = len(phase_mult) <=1
                if not shared_phases and len(phase_mult) != nb_phases :
                    return "", "", f"if the ';' syntax is used for one Lora multiplier, the multipliers for its {nb_phases} denoising phases should be specified for this multiplier"
                for phase_no, mult in enumerate(phase_mult):
                    if phase_no == 1: 
                        current_phase = phase2
                    elif phase_no == 2: 
                        current_phase = phase3
                    if "," in mult:
                        multlist = mult.split(",")
                        slist = []
                        for smult in multlist:
                            if not is_float(smult):                
                                return "", "", f"Lora sub value no {i+1} ({smult}) in Multiplier definition '{multlist}' is invalid in Phase {phase_no+1}"
                            slist.append(float(smult))
                    else:
                        if not is_float(mult):                
                            return "", "", f"Lora Multiplier no {i+1} ({mult}) is invalid"
                        slist = float(mult)
                    if shared_phases:
                        phase1[i] = phase2[i] = phase3[i] = slist
                        shared[i] = True
                    else:
                        current_phase[i] = slist
            else:
                phase1[i] = phase2[i] = phase3[i] = float(mult)
                shared[i] = True

    if merge_slist is not None:
        slists_dict["phase1"] = phase1 = merge_slist["phase1"] + phase1
        slists_dict["phase2"] = phase2 = merge_slist["phase2"] + phase2
        slists_dict["phase3"] = phase3 = merge_slist["phase3"] + phase3
        slists_dict["shared"] = shared = merge_slist["shared"] + shared

    loras_list_mult_choices_nums = [ expand_slist(slists_dict, i, num_inference_steps, model_switch_step, model_switch_step2 )  for i in range(len(phase1)) ]
    loras_list_mult_choices_nums = [ slist[0] if isinstance(slist, list) else slist for slist in loras_list_mult_choices_nums ]
    
    return  loras_list_mult_choices_nums, slists_dict, ""

def update_loras_slists(trans, slists_dict, num_inference_steps, phase_switch_step = None, phase_switch_step2 = None):
    from mmgp import offload
    sz = len(slists_dict["phase1"])
    slists = [ expand_slist(slists_dict, i, num_inference_steps, phase_switch_step, phase_switch_step2 ) for i in range(sz)  ]
    nos = [str(l) for l in range(sz)]
    offload.activate_loras(trans, nos, slists ) 



def get_model_switch_steps(timesteps, guide_phases, model_switch_phase, switch_threshold, switch2_threshold ):
    total_num_steps = len(timesteps)
    model_switch_step = model_switch_step2 = None
    for i, t in enumerate(timesteps):
        if guide_phases >=2 and model_switch_step is None and t <= switch_threshold: model_switch_step = i
        if guide_phases >=3 and model_switch_step2 is None and t <= switch2_threshold: model_switch_step2 = i                    
    if model_switch_step is None: model_switch_step = total_num_steps
    if model_switch_step2 is None: model_switch_step2 = total_num_steps
    phases_description = ""
    if guide_phases > 1:
        phases_description = "Denoising Steps: "        
        phases_description +=  f" Phase 1 = None" if model_switch_step == 0 else f" Phase 1 = 1:{ min(model_switch_step,total_num_steps) }"
        if model_switch_step < total_num_steps:                    
            phases_description += f", Phase 2 = None" if model_switch_step == model_switch_step2 else f", Phase 2 = {model_switch_step +1}:{ min(model_switch_step2,total_num_steps) }"
            if guide_phases > 2 and model_switch_step2 < total_num_steps:  
                phases_description += f", Phase 3 = {model_switch_step2 +1}:{ total_num_steps}"
    return model_switch_step, model_switch_step2, phases_description
