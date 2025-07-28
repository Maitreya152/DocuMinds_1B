# DocuMinds - Challenge 1B

Note: The GitHub link associated with this submission is currently kept private, in order to abide by the contest instructions. Though, this zip file is self sufficient. The GitHub link can be made public once asked to do so.

## Overview

This project addresses the Challenge 1B of the Adobe India Hackathon 2025, focusing on persona-driven document intelligence. The goal is to build a system that acts as an intelligent document analyst, extracting and prioritizing the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## Input Specification

*   **Document Collection:** 3-10 related PDFs (Make sure these are placed in ./input directory)
*   **Persona Definition:** Role description with specific expertise and focus areas (This will be given as an input while doing docker run)
*   **Job-to-be-Done:** Concrete task the persona needs to accomplish (This will be given as an input while doing docker run)

## Required Output

The output should be a JSON file containing:

1.  **Metadata:**
    *   Input documents
    *   Persona
    *   Job to be done
    *   Processing timestamp
2.  **Extracted Section:**
    *   Document
    *   Page number
    *   Section title
    *   Importance\_rank
3.  **Sub-section Analysis:**
    *   Document
    *   Refined Text
    *   Page Number

The name of the JSON file would be result_{Persona}_{Job To Be Done}.json, and would be stored in the ./output directory.

## Execution Details

The Docker image can be built using the following command:

```bash
docker build --platform linux/amd64 -t your_identifier .
```

Note: The build for this part will take a lot of time as it involves pytorch installation as well.

After building the image, the solution can be run using the following command:

```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none your_identifier "Persona" "Job_To_Be_Done"
```

Note: The Persona and Job_To_Be_Done are just identifiers, they can be replaced with anything as long as they are enclosed within "".

Your container should:

*   Automatically process all PDFs from `./input` directory, generating a corresponding `result_{Persona}_{Job To Be Done}.json` in `./output`.

*   Output the total and average processing time taken by the documents.
Example: ```Starting analysis for persona: Travel Planner, job: Plan a Trip of 4 days for a group of 10 college friends.
Analysis completed in 9.53 seconds for 7 PDF files```