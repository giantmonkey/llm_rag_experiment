import os
import shutil

def move_documents(input_dir, output_dir, batch_size=20):
  # Ensure output directory exists
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # List files in the input directory
  files = os.listdir(input_dir)

  # Break the loop if input directory is empty
  if not files:
    return False

  # Move up to batch_size documents
  for file in files[:batch_size]:
    src = os.path.join(input_dir, file)
    dst = os.path.join(output_dir, file)
    shutil.move(src, dst)

  print(f"Moved {min(batch_size, len(files))} files.")

  return True
