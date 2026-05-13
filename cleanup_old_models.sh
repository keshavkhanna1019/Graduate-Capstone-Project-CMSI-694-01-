#!/bin/bash
# Cleanup script to remove old broken models

echo "🧹 Cleaning up old broken models..."
echo ""

# Delete old models
if [ -f "siamese_model_v2.h5" ]; then
    rm siamese_model_v2.h5
    echo "✅ Deleted siamese_model_v2.h5 (old broken model)"
fi

if [ -f "siamese_model_v2_weights.h5" ]; then
    rm siamese_model_v2_weights.h5
    echo "✅ Deleted siamese_model_v2_weights.h5"
fi

# Optional: Delete old enrollments (uncomment if you want to start fresh)
# if [ -f "face.db" ]; then
#     rm face.db
#     echo "✅ Deleted face.db (old enrollments)"
# fi

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Restart your server"
echo "  2. Collect NEW training data (Collect Data tab)"
echo "  3. Train NEW model (Train Model tab, 100+ epochs)"
echo "  4. Enroll faces again"
echo "  5. Test recognition"
