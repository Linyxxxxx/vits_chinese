from inference import VITS_CHINESE

vits = VITS_CHINESE('logs/bert_vits/G_50000.pth')
# vits = VITS_CHINESE('../vits_chinese_original/vits_bert_model.pth')

vits.tts('腾讯云对《腾讯云账号协议》和《腾讯云服务协议》部分内容进行了更新：《腾讯云账号协议》进一步明确了同一用户可实名账号数量受限的规定，同时将《腾讯云服务协议》中关于账号的条款进行了精简；《腾讯云服务协议》补充完善了网络安全和网络秩序相关条款、丰富了知识产权和用户业务数据条款、补充了腾讯云因用户欠费停服的通知义务、降低用户欠费的违约金比例[1]，以及修订了其他部分条款。')