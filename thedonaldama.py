import praw as pr
r = pr.Reddit('donald!')
submission = r.get_submission(submission_id = "4uxdbn")
submission.replace_more_comments(limit=100, threshold=1)
comments = submission.comments
with open("ama.txt",'a') as f:
    for comment in comments:
    #iterare through boyd 
        for replies in comment.replies:
            x = replies.author 
            if(str(x) ==  "the-realDonaldTrump"):
                print( str(comment.author) + '\n' + str(comment.body) + " " + str(replies.body))
                f.write('Q: ' + str(comment.body) + '\n' + "TRUMP: " + str(replies.body) + '\n')
