# Setup Plan

## Clean up repository

1. Remove unnecessary files and directories:
   - Delete any temporary files, logs, or datasets that are not needed for the preprocessing pipeline.
   - Ensure only relevant code, configuration files, and documentation remain.

2. Remove unnecessary features from the repository:
   - Delete any scripts or modules that are not part of the RoundixReason dataset preprocessing pipeline.
   - Identify, report and remove extra features or functionalities that are not essential to the core purpose of this repository.

3. Clean up code:
   - Remove comments and print statements that are not necessary for the end user, such as debugging information, developer notes, or performance metrics.
   - Remove comments. None of them are written by a human.
   - Make sure the import order is consistent across all files (standard libraries first, then third-party libraries, then local imports).

4. Update documentation:
   - Ensure that the README.md file accurately describes the purpose of the repository, how to use the preprocessing pipeline, and any dependencies or requirements. It should be as short as humanly possible.

## To publish this repo:

1. Remove the existing origin (if any):
   ```bash
   git remote remove origin
   ```

2. Create initial commit (if not done):
   ```bash
   git add .
   git commit -m "Initial commit: RoundixReason dataset preprocessing pipeline"
   ```

3. Create GitHub repo and push:
   ```bash
   gh repo create LockedInTheSkage/RoundixReason-Preprocessing --public --source=. --push
   ```

   Or manually:
   ```bash
   git remote add origin https://github.com/LockedInTheSkage/RoundixReason-Preprocessing.git
   git push -u origin main
   ```

4. Delete this TODO.md file after setup is complete.
