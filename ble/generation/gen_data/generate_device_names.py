from pathlib import Path
input_file = "ble_device_names_10000.txt"
output_file = "BleDeviceNames.py"




if __name__ == "__main__":
    """
    lines = Path(input_file).read_text(encoding="utf-8").splitlines()
    
    # Convert to plain-text Python list format
    result = "list = [\n"
    result += ",\n".join(f'    "{line}"' for line in lines)
    result += "\n]"
    
    # Save result
    Path(output_file).write_text(result, encoding="utf-8")
    
    print("Done. Output written to", output_file)
    """