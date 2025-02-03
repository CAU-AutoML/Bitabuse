import os
import time
from threading import Thread

from openai import OpenAI

client = OpenAI()

batch_file_dir = "./gpt_batches"

progress_status = ["validating", "in_progress", "cancelling"]
finalizing_status = ["finalizing"]
fail_status = ["failed", "cancelled", "expired"]
success_status = ["completed"]

batch_range = range(11, 17)


def main():
    print("Waiting for previous batch to finish...")
    while any(b.status in progress_status for b in client.batches.list().data):
        time.sleep(30)
    print("Previous batch finished.")

    for i in batch_range:
        print("====== starting batch", i)
        path = os.path.join(batch_file_dir, f"gpt_batch_{i}.jsonl")
        try:
            with open(path, "rb") as file:
                batch_input_file = client.files.create(file=file, purpose="batch")
        except Exception as e:
            print(f"Error opening file {path}: {e}")
            continue

        batch_input_file_id = batch_input_file.id

        try:
            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"bitabuse batch_{i}"},
            )
        except Exception as e:
            print(f"Error creating batch: {e}")
            continue

        batch_id = batch.id
        print(f"batch_id of batch {i}: {batch_id}")

        print("Waiting for batch to finish...")
        try:
            batch = client.batches.retrieve(batch_id)
            while batch.status in progress_status:
                time.sleep(30)
                batch = client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"Error retrieving batch status: {e}")
            continue

        print(f"Batch {i} ({batch_id}) finished: {batch.status}")

        if batch.status in fail_status:
            raise Exception(f"Batch {i} failed: {batch.status}")

        if batch.status in finalizing_status:
            print(f"Batch finalizing. Creating thread to wait for completion.")
            t = Thread(target=complete_wait_thread, args=(client, i, batch_id))
            t.start()


def complete_wait_thread(client, batch_num, batch_id):
    output_path = os.path.join(batch_file_dir, f"gpt_batch_{batch_num}_output.jsonl")
    try:
        batch = client.batches.retrieve(batch_id)
        while batch.status in finalizing_status:
            time.sleep(30)
            batch = client.batches.retrieve(batch_id)
    except Exception as e:
        print(f"Error in thread while retrieving batch {batch_num} status: {e}")
        return

    print(f"Batch {batch_num} ({batch_id}) completed. Downloading results...")

    try:
        file_response = client.files.content(batch.output_file_id)
        with open(output_path, "w") as f:
            f.write(file_response.text)
    except Exception as e:
        print(f"Error downloading or writing batch {batch_num} results: {e}")

    print(f"Batch {batch_num} ({batch_id}) results downloaded to {output_path}")


if __name__ == "__main__":
    main()
