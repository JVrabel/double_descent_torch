#!/bin/bash

# Set variables
API_TOKEN="a22aa5018cfec6677d3cce4aec48575ef51626a4234f3260651a1443fa018e83e642874680b366677610e87e744ad27be91ecb717574ce6eefb0f01d97c0e867"

FILE_PATH="/home/jakub/projects/double_descend/double_descent_torch/runs.zip"
FILE_NAME=$(basename "$FILE_PATH")
FILE_SIZE=$(stat --printf="%s" "$FILE_PATH")
TITLE="My Private Dataset"
DESCRIPTION="Description of the dataset"
ARTICLE_TYPE="dataset"

# Step 1: Create a new article
echo "Creating new article on Figshare..."
RESPONSE=$(curl -s -X POST https://api.figshare.com/v2/account/articles \
     -H "Authorization: token $API_TOKEN" \
     -H "Content-Type: application/json" \
     -d "{\"title\": \"$TITLE\", \"description\": \"$DESCRIPTION\", \"defined_type\": \"$ARTICLE_TYPE\"}")

# Extract article ID
ARTICLE_ID=$(echo "$RESPONSE" | grep -o '"entity_id":[^,]*' | cut -d':' -f2 | tr -d ' ')
if [ -z "$ARTICLE_ID" ]; then
    echo "Failed to create article."
    exit 1
fi

echo "Article created successfully with ID: $ARTICLE_ID"

# Step 2: Initiate file upload
echo "Initiating file upload..."
UPLOAD_RESPONSE=$(curl -s -X POST https://api.figshare.com/v2/account/articles/$ARTICLE_ID/files \
     -H "Authorization: token $API_TOKEN" \
     -H "Content-Type: application/json" \
     -d "{\"name\": \"$FILE_NAME\", \"size\": $FILE_SIZE}")

# Print the response to help debug
echo "Upload response: $UPLOAD_RESPONSE"

# Extract location URL (file upload endpoint)
LOCATION_URL=$(echo "$UPLOAD_RESPONSE" | grep -o '"location":"[^"]*' | cut -d':' -f2- | tr -d '"')
if [ -z "$LOCATION_URL" ]; then
    echo "Failed to initiate file upload."
    exit 1
fi

echo "Location URL received: $LOCATION_URL"

# Step 3: Upload the file using the location URL
echo "Uploading file..."
UPLOAD_RESULT=$(curl -X PUT "$LOCATION_URL" \
     --upload-file "$FILE_PATH")

if [[ $UPLOAD_RESULT == *"404 Not Found"* ]]; then
    echo "File upload failed."
    exit 1
fi

echo "File uploaded successfully."