# Post-Testing Script

## Command to run

./run.sh -p PORT

## Instructions

1. **Find a free port**: Identify a random open port on the Mac (e.g., using `lsof -ti:PORT` or similar).
2. **Log the command**: Print the full command to the user: `./run.sh -p <free_port>`. Do NOT run it yet.
3. **Ask for confirmation**: Ask the user if they are ready to test.
4. **Print the docs URL**: After the user agrees, print `http://computer_local_ip:PORT/docs` in the chat.
5. **Run the server**: Execute the command in the background.
6. **Stop testing**: When the user says "stop" or "stop testing", kill the server process.
