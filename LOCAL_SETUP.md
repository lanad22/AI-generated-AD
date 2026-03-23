# Local Setup

## Prerequisites

- **Conda** with the `video_describe` environment (see `environment.yml`)
- **AWS credentials** with access to the S3 bucket used for video storage
- **API keys** for Gemini, OpenAI, and Google Cloud (Speech-to-Text)
- **Youtube-Downloader** service running locally or accessible via URL

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
API_KEY=<your-api-key>
GOOGLE_APPLICATION_CREDENTIALS=<path-to-google-service-account-json>
GEMINI_API_KEY=<your-gemini-api-key>
OPENAI_API_KEY=<your-openai-api-key>

S3_VIDEO_BUCKET=<s3-bucket-name>
AWS_REGION=<aws-region>
AWS_ACCESS_KEY_ID=<your-aws-access-key>
AWS_SECRET_ACCESS_KEY=<your-aws-secret-key>

YOUTUBE_DOWNLOADER_URL=http://localhost:8001
YDX_API_URL=http://localhost:4001

# Set to "true" on EC2 to auto-delete local files after results are uploaded to S3
CLEANUP_AFTER_PROCESSING=false
```

## Running the Server

```bash
conda activate video_describe
python server.py
```

The server starts on port **8000** by default.

## Dependencies

The Youtube-Downloader service must be running on the URL specified by `YOUTUBE_DOWNLOADER_URL`.
When running locally, start it on `http://localhost:8001`.

The YouDescribeX-api backend must be running on the URL specified by `YDX_API_URL`.
When running locally with `NODE_ENV=development`, it uses port **4001**:

```bash
export NODE_ENV=development
npm start
```
