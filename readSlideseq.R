library(Seurat)
library(deldir)
library(ggplot2)
library(reticulate)

datapath<-file.path('/nfs','latdata','xinyi','slide_seq.rds')
seuratsavepath<-file.path('/nfs','latdata','xinyi','slideseq','seuratresults')
slide.seq<-readRDS(datapath)
ref<-readRDS(file.path('/nfs','latdata','xinyi','mouse_hippocampus_reference.rds'))



sobj<-list()

sobj[['slideseq3000feature']]<-SCTransform(slide.seq, assay = "Spatial", ncells = 3000, verbose = FALSE,variable.features.n = 3000)
sobj[['slideseq5000feature']]<-SCTransform(slide.seq, assay = "Spatial", ncells = 3000, verbose = FALSE,variable.features.n = 5000)
sobj[['slide.seq_rv1.3']]<-SCTransform(slide.seq, assay = "Spatial", ncells = 3000, verbose = FALSE,variable.features.n = NULL,variable.features.rv.th = 1.3)

for(sobj_i in names(sobj)){
  savedir<-file.path(seuratsavepath,sobj_i)
  if(!dir.exists(savedir)){dir.create(savedir)}
  
  sobj[[sobj_i]] <- RunPCA(sobj[[sobj_i]])
  sobj[[sobj_i]] <- RunUMAP(sobj[[sobj_i]], dims = 1:30)
  sobj[[sobj_i]] <- FindNeighbors(sobj[[sobj_i]], dims = 1:30)
  sobj[[sobj_i]] <- FindClusters(sobj[[sobj_i]], resolution = 0.3, verbose = FALSE)
  plot1 <- DimPlot(sobj[[sobj_i]], reduction = "umap", label = TRUE)
  ggsave(file.path(savedir,'umap.pdf'),plot1)  
  plot2 <- SpatialDimPlot(sobj[[sobj_i]], stroke = 0)
  ggsave(file.path(savedir,'spatialID.pdf'),plot2)  
  
  anchors <- FindTransferAnchors(reference = ref, query = sobj[[sobj_i]], normalization.method = "SCT", 
                                 npcs = 50)
  predictions.assay <- TransferData(anchorset = anchors, refdata = ref$celltype, prediction.assay = TRUE, 
                                    weight.reduction = sobj[[sobj_i]][["pca"]], dims = 1:50)
  sobj[[sobj_i]][["predictions"]] <- predictions.assay
  
  DefaultAssay(sobj[[sobj_i]]) <- "SCT"
  sobj[[sobj_i]] <- FindSpatiallyVariableFeatures(sobj[[sobj_i]], assay = "SCT", slot = "scale.data", features = VariableFeatures(sobj[[sobj_i]])[1:1000], 
                                             selection.method = "moransi", x.cuts = 100, y.cuts = 100)
  splot<-SpatialFeaturePlot(sobj[[sobj_i]], features = head(SpatiallyVariableFeatures(sobj[[sobj_i]], selection.method = "moransi"), 
                                                       6), ncol = 3, alpha = c(0.1, 1), max.cutoff = "q95")
  ggsave(file.path(savedir,'spatial_varFeatures.pdf'),splot)
  
  saveRDS(sobj[[sobj_i]],file.path(savedir,paste0(sobj_i,'.rds')))
  
  varFeature<-VariableFeatures(sobj[[sobj_i]])
  saveRDS(varFeature,file.path(savedir,'varfeatures.rds'))
  write.csv(as.matrix(varFeature,ncol=1),file.path(savedir,'varfeatures.csv'))
  varSpatialFeature<-SpatiallyVariableFeatures(sobj[[sobj_i]], selection.method = "moransi")
  saveRDS(varSpatialFeature,file.path(savedir,'varfeatures_spatial.rds'))
  write.csv(as.matrix(varSpatialFeature,ncol=1),file.path(savedir,'varfeatures_spatial.csv'))
  
  ad <- Convert(from = sobj[[sobj_i]], to = "anndata", filename = file.path(savedir,paste0(sobj_i,'.h5ad')))
}

