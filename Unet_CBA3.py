class UNetCBA3(nn.Module):
    def __init__(self):
        super(UNetCBA3, self).__init__()
        bilinear = True
        self.down1=Down(3,64,7)
        self.down2=Down(64,128,7)

        self.g1 = Group(conv=default_conv, dim=256, kernel_size=3, blocks=5)
        self.up1 = Up(512, 256 ,3,bilinear)
        self.up2 = Up(256, 64 ,5, bilinear)
        self.up3=Up(128,32,7)
        self.Conv=nn.Conv2d(32,3,kernel_size=1)

    def forward(self, x):
        x300=x[0]
        x300_=self.down1(x300)
        x150=x[1]
        x150_=self.down2(x150)
        x75=x[2]   #
        x38=x[3]

        x38=self.g1(x38)


        x=self.up1(x38,x75)
        x=self.up2(x,x150_)
        x=self.up3(x,x300_)

        x=self.Conv(x)
        return x

