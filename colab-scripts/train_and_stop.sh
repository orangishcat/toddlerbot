SKILL="${1:-cartwheel}"
bash sh train_skill.sh $SKILL

curl -X POST \
  "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
