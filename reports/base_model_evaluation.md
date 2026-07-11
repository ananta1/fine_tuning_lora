# Baseline Model Evaluation Report

This report evaluates the performance of the original, un-trained base model (`unsloth/Qwen2.5-7B`) on 10 domain-specific customer support and technical questions for `practiceyourspeech.com`.

## Evaluation Table

| Question | Base Model Answer | Problem |
| :--- | :--- | :--- |
| **Q1: Why is the Subscribe button greyed out on my Profile page?** | "A greyed-out subscribe button usually indicates that you are already subscribed to a plan, or there is an issue with your account credentials. Check if you are logged in correctly or contact site support." | **Generic & Incorrect**: It does not identify our specific frontend block (`disabled={true}`) or the Stripe checkout portal maintenance issue. |
| **Q2: My payment went through, but I am still not upgraded. Why?** | "Sometimes payments take time to process. Please wait a few hours and refresh your account. If it still doesn't show, check your credit card statement or contact support." | **Generic**: It doesn't know about our backend Stripe float database serialization exception, which is the actual root cause that requires manual admin database update. |
| **Q3: My speech has been stuck in 'IN_PROGRESS' for an hour. How can I fix it?** | "Processing times vary depending on network traffic. Let it complete or try re-uploading your video." | **Misleading**: The user should run `reprocess_stuck_video.py` or restart AWS Transcribe/Rekognition jobs, not just wait indefinitely or re-upload (which uses more quota). |
| **Q4: What emotions does visual analysis track?** | "Visual analysis can track expressions like happiness, sadness, and anger." | **Incomplete**: It fails to list the exact eight emotions supported by AWS Rekognition on our platform (HAPPY, CALM, ANGRY, SAD, FEAR, SURPRISE, CONFUSED, and DISGUSTED) and how they compile to `primaryEmotion`. |
| **Q5: How is WPM calculated on the platform?** | "Words per minute is calculated by dividing total words by total duration." | **Generic**: Unsure about our exact start/end metrics (`Last Word End Time - First Word Start Time`) and optimal targets (130-150 WPM). |
| **Q6: Why is my Pronunciation Clarity score so low?** | "You might have a heavy accent, or the system did not understand your words clearly." | **Rude & Inaccurate**: It doesn't mention our system-specific `CONFIDENCE_THRESHOLD` of 0.85, nor suggest quiet environment, background noise cancellation, or external microphones. |
| **Q7: What filler words are tracked?** | "It tracks standard fillers like um, ah, and like." | **Incomplete**: It should name the exact eight words our regex compiler matches ('ah', 'uh', 'um', 'like', 'so', 'you know', 'actually', 'basically'). |
| **Q8: Why can't I see subscription pricing when enrolled in a class?** | "Pricing is usually visible in the plans tab. If not, contact your administrator." | **Incorrect**: Doesn't state that student profiles hide pricing intentionally when linked to an instructor's classroom ID. |
| **Q9: How do custom evaluation rubrics work?** | "You can set custom rubrics by typing them into the evaluation settings page." | **Incorrect**: Doesn't explain how the backend retrieves the instructor's classroom criteria override and injects it into the Amazon Bedrock template. |
| **Q10: What does the reprocess script do?** | "A reprocess script usually fixes database errors or updates records." | **Generic**: Doesn't name `reprocess_stuck_video.py`, explain AWS Transcribe/Rekognition status checks, or mention the `--force-restart` flag. |

## Baseline Observations
The baseline model performs poorly on domain-specific inquiries. It provides generic, templated responses that can mislead customers, suggest incorrect troubleshooting procedures, and ignore custom architecture parameters (such as AWS services, specific codebase bugs, and classroom structures). This confirms the necessity of SFT and DPO alignment.
