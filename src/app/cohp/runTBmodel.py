from TBmodel import COHPDashApp

app = COHPDashApp()
requests_pathname_prefix = "/Interact/"
cohp_app = app.create_cohp_dashapp(requests_pathname_prefix)

#if __name__ == '__main__':
cohp_app.run_server(debug=True)
#from TBmodel import TBModel

#direct =  "Graphene/" #final_PbO/uniform/new_new/"#"LK99/hy-pb1/uniform/"##"testing_oxi/Bi2O3/prim/"#"final_Si/uniform/no_detangle/"#"final_Si/uniform/lrealFALSE/"#"final_Graphene/uniform/"##"final_Si/uniform/7x7x7/" #'final_PbO/uniform/noTimeSym/'"PbO_fmtedUNKtest/"
#test = TBModel(direct,min_hopping_dist=10)

#BandstrucDir = "Graphene/bandstruc/" #"final_Graphene/bandstruc/"#"final_PbO/bandstruc/"#"final_Si/bandstruc/"#
#test.get_DFT_bandstruc(BandstrucDir)
#test.plot_hopping()
#test.plot_overlaps()
#test.get_bandstructure(num_kpts=20)
#test.plotBS()

#test.make_COHP_dashapp()
#test.get_COHP("BS",orbs=[{"Si":["s","p"]},{"Si":["s","p"],"Pb":["s"]}],colorhalf=15,ylim=(-12,8))

