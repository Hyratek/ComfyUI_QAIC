
import numpy as np
import qaic.session
import yaml 
import os 
import subprocess
import qaic
from comfy.cli_args import args
import platform
import sys
from enum import Enum
import psutil
import torch
import logging
import re
import subprocess
import logging
from enum import Enum
# from device_manager import QAICInferenceSession, device_type, aic_to_np_dtype_mapping
########


class DeviceType(Enum):
    DEVICE_TYPE_CPU = "cpu"
    DEVICE_TYPE_QAIC = "qaic"
    DEVICE_TYPE_GPU = "gpu"

device_type = DeviceType

class DRAMState(Enum):
    DISABLED = 0    # No DRAM present: no need to move models to DRAM
    NO_DRAM = 1   # Very low DRAM: enable all options to save DRAM
    LOW_DRAM = 2  # Low available DRAM
    NORMAL_DRAM = 3  # Normal DRAM state
    HIGH_DRAM = 4  # Sufficient DRAM available
    SHARED = 5      # DRAM shared between CPU and processing unit, but models still need to be moved

class ProcessingUnitState(Enum):
    ACCELERATOR = 0 # Processing on dedicated accelerator (Qualcomm A100)
    CPU = 1         # Processing on CPU
    HYBRID = 2      # Hybrid mode (using both CPU and accelerator)

# Determine DRAM State (equivalent to VRAM in this case)
dram_state = DRAMState.NORMAL_DRAM
set_dram_to = DRAMState.NORMAL_DRAM
processing_unit_state = ProcessingUnitState.ACCELERATOR

# DRAM information based on your card
total_dram = 15663104  # Total DRAM in KB
free_dram = 15411900   # Free DRAM in KB
dram_fragmentation = 0  # No fragmentation
#######




try:
    result = subprocess.run(["sudo", "/opt/qti-aic/tools/qaic-util", "-q"], capture_output=True, text=True, check=True)
    output = result.stdout

    free_memory = None
    for line in output.splitlines():
        if "Dram Free" in line:
            free_memory = int(line.split(":")[1].strip().split()[0])  
            break


    if free_memory is not None and free_memory < 100 * 1024:  
        raise MemoryError("Không đủ bộ nhớ DRAM!")
    else:
        OOM_EXCEPTION = None 

except MemoryError:
    OOM_EXCEPTION = MemoryError
except Exception:
    OOM_EXCEPTION = Exception  


if OOM_EXCEPTION:
    print(f"Error: {OOM_EXCEPTION}") 
else:
    print("Available Dram")




def initialize(args):
    global device
    device = args.qaic_device

# try:

#     torch_version = torch.version.__version__

# except Exception as e:
#     logging.error(f"An error occurred while checking torch version: {e}")
# logging.info(f"PyTorch version: {torch_version}")

lowdram_available = True

if args.deterministic:
    logging.info("Using deterministic algorithms is not applicable as torch is removed.")


if args.cpu:
    processing_unit_state = ProcessingUnitState.CPU

def get_comfyui_version():
    try:
        version = subprocess.check_output(['git', 'describe', '--tags'], stderr=subprocess.STDOUT).strip().decode('utf-8')
        return version
    except subprocess.CalledProcessError as e:
        logging.warning(f"Failed to get ComfyUI version: {e.output.decode('utf-8').strip()}")
        return "unknown"

comfyui_version = get_comfyui_version()
logging.info(f"ComfyUI version: {comfyui_version}")

def check_device_qualcomm():
    try:
        qaic_output = subprocess.run("sudo /opt/qti-aic/tools/qaic-util -q", 
                                     shell=True, capture_output=True, text=True).stdout
        
        lines = qaic_output.splitlines()
        device_ids = []
        for i, line in enumerate(lines):
            if line.startswith("QID"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    device_id = parts[1]
                    device_ids.append(device_id)

        results = {}
        for device_id in device_ids:
            try:
                device_output = subprocess.run(f"sudo /opt/qti-aic/tools/qaic-util -d {device_id} -q", 
                                               shell=True, capture_output=True, text=True).stdout

                if "Qualcomm Device a100" in device_output or "AIC100" in device_output:
                    results[device_id] = True
                else:
                    results[device_id] = False

            except Exception as e:
                print(f"Error while checking device {device_id}: {e}")
                results[device_id] = False

        return results

    except Exception as e:
        print(f"Error while checking devices: {e}")
        return {}


# def is_device_qaic():
#     global processing_unit_state
#     if processing_unit_state == ProcessingUnitState.ACCELERATOR:
#         results = check_device_qualcomm()
#         return any(results.values())
#     else:
#         return False



# def get_total_memory(dev=None, torch_free_too=False):
#     try:
#         # Chạy lệnh QAIC để lấy thông tin bộ nhớ
        
#         result = subprocess.run(
#             ['sudo', '/opt/qti-aic/tools/qaic-util', '-q'],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             check=True,
#             text=True
#         )
        
#         output = result.stdout

#         # Sử dụng regex để lấy thông tin Dram Total và Dram Free từ output
#         matches = re.findall(r'Dev Link:(/dev/accel/\w+).*?Dram Total:\s*(\d+)\s*KB\s+Dram Free:\s*(\d+)\s*KB', output, re.DOTALL)
        
#         if not matches:
#             raise ValueError("Device not found or no valid memory data")

#         mem_total = 0
        
#         for dev_link, dram_total, dram_free in matches:
#             try:
#                 mem_total_torch = float(dram_total)  # Chuyển đổi thành số nguyên
#                 mem_total += float(mem_total_torch)
#             except ValueError:
#                 print(f"Invalid dram_total value: {dram_total}")
#                 continue

#         # Nếu yêu cầu thêm thông tin torch_free_too, trả về giá trị cho torch
#         if torch_free_too:
#             return mem_total, mem_total_torch
#         else:
#             return float(mem_total)

#     except subprocess.CalledProcessError as e:
#         print(f"Action Error: {e}")
#         return 0 if not torch_free_too else (0, 0)

#     except Exception as e:
#         print(f"Error: {e}")
#         return 0 if not torch_free_too else (0, 0)
def get_total_memory(dev=None, torch_total_too=False):
    if dev is None or (hasattr(dev, 'type') and dev.type == 'cpu'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total  
    else:
        try:
            result = subprocess.run(
                ['sudo', '/opt/qti-aic/tools/qaic-util', '-q'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            
            output = result.stdout

            matches = re.findall(r'Dev Link:(/dev/accel/\w+).*?Dram Total:\s*(\d+)\s*KB\s+Dram Free:\s*(\d+)\s*KB', output, re.DOTALL)
            
            if not matches:
                raise ValueError("Device not found or no valid memory data")
            
            mem_total = 0
            for dev_link, dram_total, dram_free in matches:
                mem_total += int(dram_total) * 1024 
            
            mem_total_torch = mem_total 
        except subprocess.CalledProcessError as e:
            print(f"Error running qaic-util: {e}")
            return 0 if not torch_total_too else (0, 0)

    if torch_total_too:
        return mem_total, mem_total_torch
    else:
        return mem_total


import torch


def is_device_type(device, type):
    if hasattr(device, 'type'):
        if device.type == type:
            return True
    return False

def is_device_qaic(device):
    return is_device_type(device, 'qaic')

def get_device_type(device):
    if args.cpu:
        return device_type.DEVICE_TYPE_CPU
    elif is_device_qaic(device):
        return device_type.DEVICE_TYPE_QAIC
    else:
        return device_type.DEVICE_TYPE_GPU



class qaic_engine:
    def __init__(self, 
                 model_path: str,
                 dev_id = [0], 
                 activate: bool = True, 
                 enable_debug_logs: bool = False,):
        self.model_path = model_path


    #load mobel by qaic
    def load_model_qaic(self):
        qaic_engine = qaic.Session([self.model_path], dev_id=1)
        

        




def get_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = get_device_type(device)
    if is_device_type(device_type, device_type.DEVICE_TYPE_QAIC):
        return qaic_engine()
    else:
        return torch.device("cpu")

print(get_torch_device())

#if get_torch_device "qaic" then return aic_to_np_dtype_mapping dtype is numpy dtype
# if get_torch_device() == "qaic":






# try:
#     logging.info("pytorch version: {}".format(torch.version.__version__))
# except:
#     pass

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = False

VAE_DTYPES = [torch.float32]

# try:
#     if is_device_qaic(): 
#         if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
#             ENABLE_QAIC_ATTENTION = True
#         if np.dtype('bfloat16').itemsize == 2:  # Check if bfloat16 is supported in NumPy
#             VAE_DTYPES = [np.dtype('bfloat16')] + VAE_DTYPES
# except:
#     pass

ENABLE_SPLIT_CROSS_ATTENTION = False
if args.use_split_cross_attention:
    ENABLE_SPLIT_CROSS_ATTENTION = True
    logging.info("Using split cross attention for optimization.")


if args.cpu_vae:
    VAE_DTYPES = [torch.float32]


if args.lowdram:
    set_dram_to = DRAMState.LOW_DRAM
    lowddam_available = True
elif args.nodram:
    set_dram_to = DRAMState.NO_DRAM
elif args.highdram or args.dram_only:
    dram_state = DRAMState.HIGH_DRAM

FORCE_FP32 = False
FORCE_FP16 = False

if args.force_fp32:
    logging.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    logging.info("Forcing FP16.")
    FORCE_FP16 = True

if lowdram_available:
    if set_dram_to in (DRAMState.LOW_DRAM, DRAMState.NO_DRAM):
        dram_state = set_dram_to


if processing_unit_state != ProcessingUnitState.ACCELERATOR:
    dram_state = DRAMState.DISABLED

if processing_unit_state == ProcessingUnitState.HYBRID:
    dram_state = DRAMState.SHARED


logging.info(f"Set dram state to: {dram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")

######
 
#no using pytorch
def get_torch_device_name(device):
    if is_device_qaic(device):
        return "{} {} : {}".format(device, args.parser_qaic_args())       


#####

try:
    logging.info("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logging.warning("Could not pick default device.")




current_loaded_models = []


def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem

# class LoadedModel:
#     def __init__(self, model):
#         self.model_path = model
#         self.device = model.load_device
#         self.weights_loaded = False
#         self.real_model = None
#         self.currently_used = True

#     def model_memory(self):
#         return self.model.model_size()

#     def model_offloaded_memory(self):
#         return self.model.model_size() - self.model.loaded_size()

#     def model_memory_required(self, device):
#         if device == self.model.current_loaded_device():
#             return self.model_offloaded_memory()
#         else:
#             return self.model_memory()

#     def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
#         patch_model_to = self.device

#         self.model.model_patches_to(self.device)
#         self.model.model_patches_to(self.model.model_dtype())

#         load_weights = not self.weights_loaded

#         if self.model.loaded_size() > 0:
#             use_more_vram = lowvram_model_memory
#             if use_more_vram == 0:
#                 use_more_vram = 1e32
#             self.model_use_more_vram(use_more_vram)
#         else:
#             try:
#                 # Use QAICInferenceSession to load the model
#                 self.real_model = qaic.Session(
#                     model_path=self.model.model_path,
#                     dev_id= 1,
#                     activate=True,
#                     enable_debug_logs=False
#                 )
#             except Exception as e:
#                 self.model.unpatch_model(self.model.offload_device)
#                 self.model_unload()
#                 raise e

#         self.weights_loaded = True
#         return self.real_model

#     def should_reload_model(self, force_patch_weights=False):
#         if force_patch_weights and self.model.lowvram_patch_counter() > 0:
#             return True
#         return False

#     def model_unload(self, memory_to_free=None, unpatch_weights=True):
#         if memory_to_free is not None:
#             if memory_to_free < self.model.loaded_size():
#                 freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
#                 if freed >= memory_to_free:
#                     return False
#         self.model.unpatch_model(self.model.offload_device, unpatch_weights=unpatch_weights)
#         self.model.model_patches_to(self.model.offload_device)
#         self.weights_loaded = self.weights_loaded and not unpatch_weights
#         self.real_model = None
#         return True

#     def model_use_more_vram(self, extra_memory):
#         return self.model.partially_load(self.device, extra_memory)

#     def __eq__(self, other):
#         return self.model is other.model
class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.weights_loaded = False
        self.real_model = None
        self.currently_used = True

    def model_memory(self):
        return self.model.model_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        load_weights = not self.weights_loaded

        if self.model.loaded_size() > 0:
            use_more_vram = lowvram_model_memory
            if use_more_vram == 0:
                use_more_vram = 1e32
            self.model_use_more_vram(use_more_vram)
        else:
            try:
                self.real_model = self.model.patch_model(device_to=patch_model_to, lowvram_model_memory=lowvram_model_memory, load_weights=load_weights, force_patch_weights=force_patch_weights)
            except Exception as e:
                self.model.unpatch_model(self.model.offload_device)
                self.model_unload()
                raise e

        self.weights_loaded = True
        return self.real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.unpatch_model(self.model.offload_device, unpatch_weights=unpatch_weights)
        self.model.model_patches_to(self.model.offload_device)
        self.weights_loaded = self.weights_loaded and not unpatch_weights
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory):
        return self.model.partially_load(self.device, extra_memory)

    def __eq__(self, other):
        return self.model is other.model
    
#load model by qaic
def load_model_qaic(model_path):
    qaic_engine = qaic.Session([model_path], dev_id=1)
    return qaic_engine

def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_dram(extra_memory)
            if extra_memory <= 0:
                break

def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 1.2

EXTRA_RESERVED_DRAM = 200 * 1024 * 1024

if any(platform.win32_ver()):
    EXTRA_RESERVED_DRAM = 500 * 1024 * 1024

if args.reserve_dram is not None:
    EXTRA_RESERVED_DRAM = args.reserve_dram * 1024 * 1024 * 1024
    logging.debug("Reserving {}MB dram for other applications.".format(EXTRA_RESERVED_DRAM / (1024 * 1024)))\


def extra_reserved_memory():
    return EXTRA_RESERVED_DRAM

def unload_model_clones(model, unload_weights_only=True, force_unload=True):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    if len(to_unload) == 0:
        return True

    same_weights = 0
    for i in to_unload:
        if model.clone_has_same_weights(current_loaded_models[i].model):
            same_weights += 1

    if same_weights == len(to_unload):
        unload_weight = False
    else:
        unload_weight = True

    if not force_unload:
        if unload_weights_only and unload_weight == False:
            return None
    else:
        unload_weight = True

    for i in to_unload:
        logging.debug("unload clone {} {}".format(i, unload_weight))
        current_loaded_models.pop(i).model_unload(unpatch_weights=unload_weight)

    return unload_weight

def free_memory(memory_required, device, keep_loaded=[]):
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) -1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            # Add your logic here
            pass

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not DISABLE_SMART_MEMORY:
            # Add your logic here
            pass
        logging.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
        if current_loaded_models[i].model_unload(memory_to_free):
            # Add your logic here
            pass

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache(device)
    else:
        if dram_state != DRAMState.HIGH_DRAM:
            # Add your logic here
            pass
    return unloaded_models

def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    global dram_state
    
    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models = set(models)
    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)
        loaded = None

        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            if loaded.should_reload_model(force_patch_weights=force_patch_weights): #TODO: cleanup this model reload logic
                current_loaded_models.pop(loaded_model_index).model_unload(unpatch_weights=True)
                loaded = None
            else:
                loaded.currently_used = True
                models_already_loaded.append(loaded)

        if loaded is None:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem + offloaded_memory(models_already_loaded, d), d, models_already_loaded)
                free_mem = get_free_memory(d)
                if free_mem < minimum_memory_required:
                    logging.info("Unloading models for lowram load.") #TODO: partial model unloading when this case happens, also handle the opposite case where models can be unlowvramed.
                    models_to_load = free_memory(minimum_memory_required, d)
                    logging.info("{} models unloaded.".format(len(models_to_load)))
                else:
                    use_more_memory(free_mem - minimum_memory_required, models_already_loaded, d)
        if len(models_to_load) == 0:
            return

    logging.info(f"Loading {len(models_to_load)} new model{'s' if len(models_to_load) > 1 else ''}")

    total_memory_required = {}

    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model, unload_weights_only=True, force_unload=False) #unload clones where the weights are different
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for loaded_model in models_already_loaded:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for loaded_model in models_to_load:
        weights_unloaded = unload_model_clones(loaded_model.model, unload_weights_only=False, force_unload=False) #unload the rest of the clones where the weights can stay loaded
        if weights_unloaded is not None:
            loaded_model.weights_loaded = not weights_unloaded

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.1 + extra_mem, device, models_already_loaded)       

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            dram_set_state = DRAMState.DISABLED
        else:
            dram_set_state = dram_state
        lowdram_model_memory = 0
        if lowdram_available and (dram_set_state == DRAMState.LOW_DRAM or dram_set_state == DRAMState.NORMAL_DRAM) and not force_full_load:
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowdram_model_memory = max(64 * (1024 * 1024), (current_free_mem - minimum_memory_required), min(current_free_mem * 0.4, current_free_mem - minimum_inference_memory()))
            if model_size <= lowdram_model_memory: #only switch to lowvram if really necessary
                lowdram_model_memory = 0

        if dram_set_state == DRAMState.NO_DRAM:
            lowdram_model_memory = 64 * 1024 * 1024

        cur_loaded_model = loaded_model.model_load(lowdram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)

    devs = set(map(lambda a: a.device, models_already_loaded))
    for d in devs:
        if d != torch.device("cpu"):
            free_mem = get_free_memory(d)
            if free_mem > minimum_memory_required:
                use_more_memory(free_mem - minimum_memory_required, models_already_loaded, d)
    return

def load_model_gpu(model):
    return load_models_gpu([model])
    # qaic.Session([model], dev_id=1)

def loaded_models(only_currently_used=False):
    output = []
    for m in current_loaded_models:
        if only_currently_used:
            if not m.currently_used:
                continue

        output.append(m.model)
    return output

def cleanup_models(keep_clone_weights_loaded=False):
    to_delete = []
    for i in range(len(current_loaded_models)):
        #TODO: very fragile function needs improvement
        num_refs = sys.getrefcount(current_loaded_models[i].model)
        if num_refs <= 2:
            if not keep_clone_weights_loaded:
                to_delete = [i] + to_delete
            #TODO: find a less fragile way to do this.
            elif sys.getrefcount(current_loaded_models[i].real_model) <= 3: #references from .real_model + the .model
                to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x

def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except: #Old pytorch doesn't have .itemsize
            pass
    return dtype_size

def unet_offload_device():
    if dram_state == DRAMState.HIGH_DRAM:
        return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    torch_dev = torch.device("cpu")
    if dram_state == DRAMState.HIGH_DRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev

# def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
#     if args.bf16_unet:
#         return torch.bfloat16
#     if args.fp16_unet:
#         return torch.float16
#     if args.fp8_e4m3fn_unet:
#         return torch.float8_e4m3fn
#     if args.fp8_e5m2_unet:
#         return torch.float8_e5m2

#     fp8_dtype = None
#     try:
#         for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
#             if dtype in supported_dtypes:
#                 fp8_dtype = dtype
#                 break
#     except:
#         pass

#     if fp8_dtype is not None:
#         free_model_memory = maximum_vram_for_weights(device)
#         if model_params * 2 > free_model_memory:
#             return fp8_dtype

#     for dt in supported_dtypes:
#         if dt == np.float16 and should_use_fp16(device=device, model_params=model_params):
#             if np.float16 in supported_dtypes:
#                 return np.float16
#         if dt == np.dtype('bfloat16') and np.dtype('bfloat16').itemsize == 2:
#             if np.dtype('bfloat16') in supported_dtypes:
#                 return np.dtype('bfloat16')

#     for dt in supported_dtypes:
#         if dt == np.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
#             if np.float16 in supported_dtypes:
#                 return np.float16
#         if dt == np.dtype('bfloat16') and should_use_bf16(device, model_params=model_params, manual_cast=True):
#             if np.dtype('bfloat16') in supported_dtypes:
#                 return np.dtype('bfloat16')

#     return np.float32

def maximum_dram_for_weights(device):
    total_memory = get_total_memory()

    # Đảm bảo total_memory là một số nguyên hoặc số thực
    if not isinstance(total_memory, (int, float)):
        raise TypeError(f"Expected a numeric value from get_total_memory, but got {type(total_memory)}")

    # Tính toán bộ nhớ khả dụng cho mô hình
    return (total_memory * 0.88 - minimum_inference_memory())



def unet_dtype(device="cpu", model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    if model_params < 0:
        model_params = 1000000000000000000000
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2

    fp8_dtype = None
    try:
        for dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            if dtype in supported_dtypes:
                fp8_dtype = dtype
                break
    except:
        pass

    if fp8_dtype is not None:
        free_model_memory = maximum_dram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return fp8_dtype

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device="cpu", model_params=model_params):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device="cpu", model_params=model_params):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device="cpu", model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device="cpu", model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32


def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=[ torch.float16, torch.bfloat16, torch.float32]):
    if weight_dtype == torch.float32:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16

    return torch.float32

def text_encoder_offload_device():
    if args.dram_only:
        return get_torch_device()
    else:
        return torch.device("cpu")
    
def text_encoder_device():
    if args.dram_only:
        return get_torch_device()
    elif dram_state == DRAMState.HIGH_DRAM or dram_state == DRAMState.NORMAL_DRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")
    
def text_encoder_initial_device(load_device, offload_device, model_size=0):
    if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
        # return offload_devices
        return torch.device("cpu")

    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    else:
        return offload_device
    
def text_encoder_dtype(device=None):
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    elif args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    elif args.fp16_text_enc:
        return torch.float16
    elif args.fp32_text_enc:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16

    return torch.float16

def intermediate_device():
    if args.dram_only:
        return get_torch_device()
    else:
        return torch.device("cpu")
    
def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def vae_offload_device():
    if args.dram_only:
        return get_torch_device()
    else:
        return torch.device("cpu")
    
def vae_dtype(device="cpu", allowed_dtypes=[]):
    global VAE_DTYPES
    if args.fp16_vae:
        return torch.float16
    elif args.bf16_vae:
        return torch.bfloat16
    elif args.fp32_vae:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d

    return VAE_DTYPES[0]

def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "qaic"

def supports_dtype(device, dtype): #TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False

def supports_cast(device, dtype): #TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True

    return False

def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype

def device_supports_non_blocking(device):
    if is_device_cpu(device):
        return False
    if args.dram_only:
        return False
    if is_device_qaic(device):
        return False
    if args.deterministic: #TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    return True

def device_should_use_non_blocking(device):
    if not device_supports_non_blocking(device):
        return False
    return False
    # return True #TODO: figure out why this causes memory issues on Nvidia and possibly others

def force_channels_last():
    if args.force_channels_last:
        return True

    #TODO
    return False

def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        return weight.to(dtype=dtype, copy=copy)

    r = torch.empty_like(weight, dtype=dtype, device=device)
    r.copy_(weight, non_blocking=non_blocking)
    return r

def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)

# def cast_to_device(tensor, device ,  dtype, copy=False):
#     device_supports_cast = False
#     if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
#         device_supports_cast = True
#     elif tensor.dtype == torch.bfloat16:
#         if is_device_qaic(device):
#             device_supports_cast = True

#     non_blocking = device_should_use_non_blocking(device)

#     if device_supports_cast:
#         if copy:
#             if tensor.device == device:
#                 return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
#             return tensor.to(device, copy=copy, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
#         else:
#             return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
#     else:
#         return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        #TODO: more reliable way of checking for flash attention?
        if is_device_qaic():
            return True
    return False

def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention
    try:
        macos_version = tuple(int(n) for n in platform.mac_ver()[0].split("."))
        if (14, 5) <= macos_version < (14, 7):  # black image bug on recent versions of MacOS
            upcast = True
    except:
        pass
    if upcast:
        return torch.float32
    else:
        return None


def get_free_memory(device, torch_free_too=True):
    try:
        result = subprocess.run(
            ['sudo', '/opt/qti-aic/tools/qaic-util', '-q'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        
        output = result.stdout

        matches = re.findall(r'Dev Link:(/dev/accel/\w+).*?Dram Total:\s*(\d+)\s*KB\s+Dram Free:\s*(\d+)\s*KB', output, re.DOTALL)
        
        if not matches:
            raise ValueError("Device not found")
        
        mem_free_total = 0
        
        for dev_link, dram_total, dram_free in matches:
            mem_free_torch = int(dram_free)
            mem_free_total += mem_free_torch  

        if torch_free_too:
            return mem_free_total, mem_free_torch
        else:
            return mem_free_total

    except subprocess.CalledProcessError as e:
        print(f"Action Eror : {e}")
        return 0 if not torch_free_too else (0, 0)

    except Exception as e:
        print(f"Error: {e}")
        return 0 if not torch_free_too else (0, 0)

def maximum_dram_for_weights(device="cpu"):
    return (get_total_memory(device) * 0.88 - minimum_inference_memory())


    

def cpu_mode():
    global processing_unit_state
    return processing_unit_state == ProcessingUnitState.CPU

def hybrid_mode():
    global processing_unit_state
    return processing_unit_state == ProcessingUnitState.HYBRID

def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, 'cpu')

def is_device_mps(device):
    return is_device_type(device, 'hybird')


def is_device_qaic(device):
    return is_device_type(device, 'qaic')
 
# def should_use_fp16(device="cpu", model_params=0, prioritize_performance=True, manual_cast=False):
#     if device is not None and is_device_cpu(device):
#         return True

#     if FORCE_FP16:
#         return True

#     if device is not None and is_device_qaic(device):
#         return True

#     if FORCE_FP32:
#         return False

#     if cpu_mode():
#         return False

#     if hybrid_mode():
#         return False

#     if manual_cast:
#         free_model_memory = maximum_dram_for_weights(device)
#         if (not prioritize_performance) or model_params * 4 > free_model_memory:
#             return True

#     return False

def should_use_fp16(device="cpu", model_params=0, prioritize_performance=True, manual_cast=False):
    return True





def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None and is_device_cpu(device):
        return False

    if device is not None and is_device_qaic(device):
        return True

    if FORCE_FP32:
        return False

    if cpu_mode():
        return False

    if hybrid_mode():
        return False


    # if torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 0):
    #     return True

    return False

def supports_fp8_compute(device=None):
    return False

def soft_empty_cache(device, force=False):
    if is_device_cpu(device):
        # Add your logic here for CPU device
        pass
    elif is_device_qaic(device):
        # Add your logic here for QAIC device
        pass

def unload_all_models():
    free_memory(1e30, get_torch_device())

def resolve_lowdram_weight(weight, model, key): #TODO: remove
    print("WARNING: The comfy.model_management.resolve_lowdram_weight function will be removed soon, please stop using it.")
    return weight

#TODO: might be cleaner to put this somewhere else
import threading

class InterruptProcessingException(Exception):
    pass

interrupt_processing_mutex = threading.RLock()

interrupt_processing = False
def interrupt_current_processing(value=True):
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    global interrupt_processing
    global interrupt_processing_mutex
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()