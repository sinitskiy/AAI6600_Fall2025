from chatbot_pipeline import mock_classify_conversation

conv = [{'role':'USER','message':'I have panic attacks and I live in Raleigh, NC. I have Medicaid insurance.'}]
print('CALLING mock_classify_conversation...')
try:
	res = mock_classify_conversation(conv)
	print('RESULT:')
	print(res)
except Exception as e:
	print('ERROR when calling mock_classify_conversation:', e)
	import traceback
	traceback.print_exc()
